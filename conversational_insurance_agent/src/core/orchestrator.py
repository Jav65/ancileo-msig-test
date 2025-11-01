from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from groq import Groq

from ..config import get_settings
from ..state.session_store import ConversationSessionStore
from ..utils.logging import logger
from .tooling import ToolSpec


TOOL_INSTRUCTION = (
    "You have access to specialized tools.\n"
    "If tools are required, respond ONLY with a JSON object that matches:\n"
    '{"actions": [{"tool": "tool_name", "input": { ... }}]}\n'
    "List every required tool in execution order within the array.\n"
    "After receiving tool results, use a concise_and_recommend reasoning step before replying,"
    " and always cite sources from policy documents when answering directly."
)


class ConversationalOrchestrator:
    def __init__(self, tools: List[ToolSpec]) -> None:
        self._settings = get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)
        self._session_store = ConversationSessionStore()
        self._tool_map = {tool.name: tool for tool in tools}

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

        tool_runs: List[Dict[str, Any]] = []
        max_rounds = 6
        turn = 0

        while turn < max_rounds:
            turn += 1
            reply = await self._generate_response(messages)
            parsed = self._try_parse_json(reply)
            actions = self._extract_actions(parsed)

            if actions:
                messages.append({"role": "assistant", "content": reply})

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
                        self._session_store.append_message(session_id, "assistant", failure_reply)
                        return {
                            "reply": failure_reply,
                            "tool_used": None,
                            "tool_result": None,
                            "tool_runs": tool_runs,
                        }

                    tool_input = action.get("input", {})
                    logger.info(
                        "orchestrator.tool_call",
                        session_id=session_id,
                        tool=tool_name,
                        payload=tool_input,
                        sequence=index,
                        total=len(actions),
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

            if parsed is not None and "actions" in parsed and not actions:
                logger.info(
                    "orchestrator.no_tool_actions",
                    session_id=session_id,
                    turn=turn,
                )
                messages.append({"role": "assistant", "content": reply})
                continue

            # no actions detected; treat reply as the final assistant message
            self._session_store.append_message(session_id, "assistant", reply)
            last_tool = tool_runs[-1]["name"] if tool_runs else None
            last_result = tool_runs[-1]["result"] if tool_runs else None
            return {
                "reply": reply,
                "tool_used": last_tool,
                "tool_result": last_result,
                "tool_runs": tool_runs,
            }

        logger.error(
            "orchestrator.max_rounds_exceeded",
            session_id=session_id,
            max_rounds=max_rounds,
        )
        failure_reply = (
            "I'm sorry, I'm having trouble completing that request right now. "
            "Let's try again in a moment."
        )
        self._session_store.append_message(session_id, "assistant", failure_reply)
        return {
            "reply": failure_reply,
            "tool_used": None,
            "tool_result": None,
            "tool_runs": tool_runs,
        }

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._settings.groq_model,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
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
