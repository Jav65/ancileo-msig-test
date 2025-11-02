from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, date, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from groq import Groq

from ..config import get_settings
from ..state.client_context import ClientDatum, select_preferred_trip
from ..state.session_store import ConversationSessionStore
from ..utils.logging import logger
from .profile_guidance import compose_profile_guidance
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
        clients = self._session_store.get_clients(session_id)
        guidance = compose_profile_guidance(clients)

        risk_payload = await self._maybe_prime_travel_risk_prediction(session_id, clients)

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
                f"{TOOL_INSTRUCTION}\n\n"
                "Pricing guidance: the Ancileo pricing API is unavailable. Thoughtfully derive premiums "
                "from the traveller's context and clearly explain your assumptions.\n"
                "Fulfilment guidance: once payment_status confirms a successful checkout, "
                "invoke travel_insurance_purchase with the quoteId/offer information shared in the conversation to issue the policy."
            ),
        }

        if guidance:
            logger.info(
                "orchestrator.profile_guidance",
                session_id=session_id,
                status=guidance.status,
            )
            system_message["content"] += "\n\n" + guidance.summary_text

        if risk_payload:
            system_message["content"] += "\n\n" + self._render_risk_summary(risk_payload)

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
                        self._session_store.apply_payment_context(session_id, tool_input)
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
                            self._session_store.append_message(session_id, "assistant", guard_reply)
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

        if not output_text.strip():
            fallback_reply = self._compose_tool_fallback_reply(tool_runs)
            if fallback_reply:
                output_text = fallback_reply

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

    async def _maybe_prime_travel_risk_prediction(
        self,
        session_id: str,
        clients: List[ClientDatum],
    ) -> Optional[Dict[str, Any]]:
        tool_spec = self._tool_map.get("travel_risk_prediction")
        if tool_spec is None or not clients:
            return None

        inputs = self._extract_risk_inputs(clients)
        if not inputs:
            return None

        cached = self._session_store.get_tool_result(session_id, "travel_risk_prediction")
        if isinstance(cached, dict) and cached.get("inputs") == inputs:
            return cached

        result = await tool_spec.arun(**inputs)
        payload = {
            "inputs": inputs,
            "result": result,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        self._session_store.set_tool_result(session_id, "travel_risk_prediction", payload)
        logger.info(
            "orchestrator.travel_risk_prediction.ran",
            session_id=session_id,
            destination=inputs.get("destination"),
            status=result.get("status") if isinstance(result, dict) else "unknown",
        )
        return payload

    def _extract_risk_inputs(self, clients: List[ClientDatum]) -> Optional[Dict[str, Any]]:
        for client in clients:
            trip = select_preferred_trip(client)
            if trip is None or not trip.destination:
                continue

            payload: Dict[str, Any] = {"destination": trip.destination}

            activity = (trip.metadata or {}).get("activity")
            if not activity and client.interests:
                activity = client.interests[0]
            if activity:
                payload["activity"] = activity

            if trip.start_date:
                payload["departure_date"] = trip.start_date.isoformat()
                payload["month"] = trip.start_date.strftime("%b")

            if client.personal_info.date_of_birth:
                payload["date_of_birth"] = client.personal_info.date_of_birth.isoformat()
                age = self._compute_age(
                    client.personal_info.date_of_birth,
                    trip.start_date,
                )
                if age is not None:
                    payload["age"] = age

            return payload
        return None

    @staticmethod
    def _compute_age(dob: Optional[date], reference: Optional[date]) -> Optional[int]:
        if dob is None:
            return None
        ref = reference or date.today()
        years = ref.year - dob.year
        if (ref.month, ref.day) < (dob.month, dob.day):
            years -= 1
        return max(1, years)

    def _render_risk_summary(self, payload: Dict[str, Any]) -> str:
        result = payload.get("result")
        if not isinstance(result, dict):
            return "[Claims Risk Forecast]\nstatus: unavailable"

        prediction = result.get("prediction", {})
        input_payload = result.get("input", {})
        model_state = result.get("model_state") or {}
        status = result.get("status", "unknown")

        probability_text = self._format_probability(prediction.get("claim_probability"))
        expected_amount_text = self._format_currency(prediction.get("expected_amount"))

        lines = ["[Claims Risk Forecast]"]
        lines.append(f"status: {status}")

        destination = prediction.get("destination") or input_payload.get("destination")
        if destination:
            lines.append(f"destination: {destination}")

        month = prediction.get("month") or input_payload.get("month")
        if month:
            lines.append(f"travel_month: {month}")

        if input_payload.get("age"):
            lines.append(f"traveller_age: {input_payload['age']}")

        activity = input_payload.get("activity")
        if activity:
            lines.append(f"primary_activity: {activity}")

        if probability_text:
            lines.append(f"claim_probability: {probability_text}")

        if expected_amount_text:
            lines.append(f"expected_claim_amount: {expected_amount_text}")

        refreshed_at = model_state.get("refreshed_at") if isinstance(model_state, dict) else None
        claim_rows = model_state.get("claim_rows") if isinstance(model_state, dict) else None
        if refreshed_at:
            lines.append(f"model_refreshed_at: {refreshed_at}")
        if claim_rows:
            lines.append(f"claim_records_used: {claim_rows}")

        if payload.get("generated_at"):
            lines.append(f"generated_at: {payload['generated_at']}")

        lines.append("source: travel_risk_prediction tool")

        return "\n".join(lines)

    @staticmethod
    def _format_probability(value: Any) -> Optional[str]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return f"{numeric:.1%}"

    @staticmethod
    def _format_currency(value: Any) -> Optional[str]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return f"${numeric:,.2f}"

    @staticmethod
    def _compose_tool_fallback_reply(tool_runs: List[Dict[str, Any]]) -> str:
        if not tool_runs:
            return ""

        last_run = tool_runs[-1]
        tool_name = last_run.get("name")
        result = last_run.get("result")

        if tool_name == "policy_research":
            return ConversationalOrchestrator._compose_policy_research_summary(result)

        return ""

    @staticmethod
    def _compose_policy_research_summary(result: Any) -> str:
        if not isinstance(result, dict):
            return ""

        products = result.get("products")
        reasoning = result.get("reasoning")

        lines: List[str] = []

        if isinstance(products, list) and products:
            lines.append("Here is what I can confirm from the policy taxonomy:")
            for product_entry in products:
                if not isinstance(product_entry, dict):
                    continue

                product_name = str(product_entry.get("product") or "Unnamed product").strip()
                tier = str(product_entry.get("tier") or "").strip()
                header = f"{product_name} ({tier})" if tier else product_name
                lines.append(f"- {header}")

                benefits = product_entry.get("benefits")
                if not isinstance(benefits, list):
                    continue

                for benefit in benefits:
                    if not isinstance(benefit, dict):
                        continue

                    benefit_name = str(benefit.get("name") or "Benefit").strip()
                    why = str(benefit.get("why_eligible") or "").strip()

                    detail_fragments: List[str] = []
                    parameters = benefit.get("parameters")
                    if isinstance(parameters, dict):
                        coverage_limit = parameters.get("coverage_limit") or parameters.get("limit")
                        if coverage_limit:
                            detail_fragments.append(f"limit {coverage_limit}")
                    conditions = benefit.get("conditions")
                    if isinstance(conditions, list):
                        rendered_conditions = [
                            str(item).strip() for item in conditions if str(item).strip()
                        ]
                        if rendered_conditions:
                            detail_fragments.append("conditions: " + "; ".join(rendered_conditions))

                    detail_suffix = ""
                    if detail_fragments:
                        detail_suffix = f" ({'; '.join(detail_fragments)})"

                    if why:
                        lines.append(f"  - {benefit_name}: {why}{detail_suffix}")
                    else:
                        lines.append(f"  - {benefit_name}{detail_suffix}")

        if isinstance(reasoning, str) and reasoning.strip():
            if lines:
                lines.append("")
            lines.append(reasoning.strip())

        if not lines:
            return ""

        lines.append("")
        lines.append("Source: policy taxonomy dataset.")

        return "\n".join(lines).strip()

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
                    field_label = missing[0]
                    return (
                        "I'm almost ready to set up payment?could you share "
                        f"the {field_label.lower()} so I can proceed?"
                    )

                fields_text = ", ".join(missing[:-1]) + f" and {missing[-1]}"
                return (
                    "I still need a few details before the payment step: "
                    f"{fields_text}. Once you share them, I can prepare checkout."
                )

            return (
                "I still need a required detail before the payment step. "
                "Let me know once it's available so I can prepare checkout."
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
