from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from groq import Groq

from ..config import get_settings
from ..state.client_context import ClientDatum
from ..state.session_store import ConversationSessionStore
from ..utils.logging import logger
from .tooling import ToolSpec


TOOL_INSTRUCTION = (
    "You have access to specialized tools.\n"
    "Respond using a JSON object shaped as:\n"
    '{"output": "<assistant reply or empty string>", "actions": [{"tool": "tool_name", "input": { ... }}]}\n'
    "List every required tool in execution order inside the actions array.\n"
    "When you need to call tools, set `output` to an empty string and populate `actions`.\n"
    "After tool results are available, produce the final answer by setting `output` and an empty `actions` array.\n"
    "Always cite policy sources in `output` when giving direct answers."
)


class ConversationalOrchestrator:
    def __init__(self, tools: List[ToolSpec]) -> None:
        self._settings = get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)
        self._session_store = ConversationSessionStore()
        self._tool_map = {tool.name: tool for tool in tools}

    def merge_clients(self, session_id: str, clients: List[ClientDatum], source: Optional[str] = None) -> None:
        self._session_store.merge_clients(session_id, clients, source)

    async def handle_message(
        self, *, session_id: str, user_message: str, channel: str
    ) -> Dict[str, Any]:
        conversation = self._session_store.get(session_id)
        history = conversation.get("messages", [])

        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description} | Schema: {json.dumps(tool.schema)}"
            for tool in self._tool_map.values()
        )

        system_message = {
            "role": "system",
            "content": (
                "You are Aurora, an empathetic travel insurance concierge. Don't sounds like AI!"
                "Adapt tone to the user's emotion, maintain concise yet thorough answers, "
                "and always explain reasoning with citations when referencing policies.\n\n"
                f"Channel: {channel}.\n"
                f"Available Tools:\n{tool_descriptions}\n\n"
                f"{TOOL_INSTRUCTION}"
            ),
        }

        messages: List[Dict[str, str]] = [system_message, *history]
        messages.append({"role": "user", "content": user_message})

        logger.info("orchestrator.invoke", channel=channel, session_id=session_id)

        self._session_store.append_message(session_id, "user", user_message)
        self._session_store.try_mark_verification(session_id, user_message)

        tool_runs: List[Dict[str, Any]] = []
        max_rounds = 6
        turn = 0

        while turn < max_rounds:
            turn += 1
            reply = await self._generate_response(messages)
            payload = self._coerce_json_payload(
                reply=reply,
                session_id=session_id,
                turn=turn,
            )
            actions = self._extract_actions(payload)
            payload["actions"] = actions
            payload["output"] = self._normalize_output(payload.get("output"))
            serialized_payload = json.dumps(payload, ensure_ascii=False)

            if actions:
                messages.append({"role": "assistant", "content": serialized_payload})

                for index, action in enumerate(actions, start=1):
                    tool_name = action.get("tool")
                    if not tool_name:
                        logger.warning(
                            "orchestrator.tool_missing_name",
                            session_id=session_id,
                            action=action,
                        )
                        continue

                    tool_spec = self._tool_map.get(tool_name)
                    if not tool_spec:
                        logger.error(
                            "orchestrator.unknown_tool",
                            session_id=session_id,
                            tool=tool_name,
                        )
                        failure_reply = (
                            "I'm sorry, I can't access the requested capability right now. "
                            "Could you try rephrasing your question?"
                        )
                        return self._finalize_response(
                            session_id=session_id,
                            output=failure_reply,
                            actions=[],
                            tool_runs=tool_runs,
                        )

                    tool_input = action.get("input", {})
                    logger.info(
                        "orchestrator.tool_call",
                        session_id=session_id,
                        tool=tool_name,
                        payload=tool_input,
                        sequence=index,
                        total=len(actions),
                    )

                    if tool_name == "payment_checkout":
                        readiness = self._session_store.evaluate_payment_readiness(session_id)
                        if readiness.get("status") != "ready":
                            logger.info(
                                "orchestrator.payment_guard_block",
                                session_id=session_id,
                                status=readiness.get("status"),
                            )
                            guard_reply = self._compose_payment_guard_reply(readiness)
                            if readiness.get("status") == "unverified":
                                self._session_store.request_verification(
                                    session_id,
                                    readiness.get("client_id"),
                                    readiness.get("fields", {}),
                                )
                            return self._finalize_response(
                                session_id=session_id,
                                output=guard_reply,
                                actions=[],
                                tool_runs=tool_runs,
                            )

                    tool_result = await tool_spec.arun(**tool_input)
                    tool_call_id = action.get("tool_call_id") or f"toolcall-{uuid4().hex}"
                    tool_message = {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                        "tool_call_id": tool_call_id,
                    }

                    self._session_store.set_tool_result(session_id, tool_name, tool_result)
                    messages.append(tool_message)
                    tool_runs.append(
                        {
                            "name": tool_name,
                            "input": tool_input,
                            "result": tool_result,
                            "tool_call_id": tool_call_id,
                        }
                    )

                # loop to let the assistant incorporate tool outputs
                continue

            return self._finalize_response(
                session_id=session_id,
                output=payload.get("output", ""),
                actions=actions,
                tool_runs=tool_runs,
            )

        logger.error(
            "orchestrator.max_rounds_exceeded",
            session_id=session_id,
            max_rounds=max_rounds,
        )
        failure_reply = (
            "I'm sorry, I'm having trouble completing that request right now. "
            "Let's try again in a moment."
        )
        return self._finalize_response(
            session_id=session_id,
            output=failure_reply,
            actions=[],
            tool_runs=tool_runs,
        )

    def _finalize_response(
        self,
        *,
        session_id: str,
        output: str,
        actions: Optional[List[Dict[str, Any]]],
        tool_runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized_actions = list(actions or [])
        output_text = output if isinstance(output, str) else str(output)
        payload = {"output": output_text, "actions": normalized_actions}
        self._session_store.append_message(
            session_id,
            "assistant",
            json.dumps(payload, ensure_ascii=False),
        )

        last_tool = tool_runs[-1]["name"] if tool_runs else None
        last_result = tool_runs[-1]["result"] if tool_runs else None

        return {
            "output": output_text,
            "actions": normalized_actions,
            "tool_used": last_tool,
            "tool_result": last_result,
            "tool_runs": tool_runs,
        }

    @staticmethod
    def _compose_payment_guard_reply(readiness: Dict[str, Any]) -> str:
        status = readiness.get("status")

        if status == "missing_clients":
            return (
                "Before we can secure a policy, I need the traveller's profile - "
                "name, contacts, passport and trip itinerary. "
                "Please share those details, or pass them through the integration payload."
            )

        if status == "missing_fields":
            missing = readiness.get("missing", [])
            if isinstance(missing, list) and missing:
                if len(missing) == 1:
                    fields_text = missing[0]
                else:
                    fields_text = ", ".join(missing[:-1]) + f" and {missing[-1]}"
            else:
                fields_text = "some required fields"
            return (
                "I still need a few details before the payment step: "
                f"{fields_text}. Once you share them, I can prepare checkout."
            )

        if status == "unverified":
            fields = readiness.get("fields", {})
            lines = []
            label_map = {
                "name": "Name",
                "destination": "Destination",
                "trip_type": "Trip type",
                "trip_cost": "Trip cost",
                "travel_dates": "Travel dates",
                "email_address": "Email",
                "phone_number": "Phone",
                "passport_number": "Passport number",
            }
            for key, label in label_map.items():
                value = fields.get(key)
                if value:
                    lines.append(f"- {label}: {value}")
            summary = "\n".join(lines) if lines else "- Traveller details on file"
            return (
                "Let's double-check the traveller info before payment:\n"
                f"{summary}\n"
                "Please confirm everything is correct (a simple 'Confirmed' works) so I can continue."
            )

        return (
            "I need a complete and confirmed traveller profile before creating the checkout link. "
            "Could you review the details and update anything that's missing?"
        )

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._settings.groq_model,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        message = choice.message
        return message.content.strip() if message and message.content else ""

    @staticmethod
    def _try_parse_json(payload: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
        
    @staticmethod
    def _extract_actions(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payload:
            return []

        if "actions" in payload and isinstance(payload["actions"], list):
            return [action for action in payload["actions"] if isinstance(action, dict)]

        if payload.get("action"):
            return [
                {
                    "tool": payload.get("action"),
                    "input": payload.get("input", {}),
                    "tool_call_id": payload.get("tool_call_id"),
                }
            ]

        return []

    def _coerce_json_payload(
        self,
        *,
        reply: str,
        session_id: str,
        turn: int,
    ) -> Dict[str, Any]:
        if not reply:
            return {"output": "", "actions": []}

        parsed = self._try_parse_json(reply)

        if parsed is None:
            logger.warning(
                "orchestrator.non_json_reply_coerced",
                session_id=session_id,
                turn=turn,
                reply_preview=reply[:200],
            )
            return {"output": reply, "actions": []}

        if not isinstance(parsed, dict):
            logger.warning(
                "orchestrator.unexpected_json_type",
                session_id=session_id,
                turn=turn,
                received_type=type(parsed).__name__,
            )
            if isinstance(parsed, str):
                return {"output": parsed, "actions": []}

            try:
                serialized = json.dumps(parsed, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized = str(parsed)

            return {"output": serialized, "actions": []}

        return dict(parsed)

    @staticmethod
    def _normalize_output(value: Any) -> str:
        if isinstance(value, str):
            return value

        if value is None:
            return ""

        try:
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            pass

        return str(value)
