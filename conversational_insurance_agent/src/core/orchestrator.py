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
    "If a tool is required, respond ONLY and ONLY with a JSON object that matches:\n"
    '{"action": "tool_name", "input": { ... }}\n'
    "Always cite sources from policy documents when answering directly."
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

        reply = await self._generate_response(messages)
        parsed = self._try_parse_json(reply)

        self._session_store.append_message(session_id, "user", user_message)

        if parsed and parsed.get("action") in self._tool_map:
            tool_name = parsed["action"]
            tool_input = parsed.get("input", {})
            tool_spec = self._tool_map[tool_name]
            logger.info(
                "orchestrator.tool_call",
                session_id=session_id,
                tool=tool_name,
                payload=tool_input,
            )
            tool_result = await tool_spec.arun(**tool_input)
            tool_call_id = parsed.get("tool_call_id") or f"toolcall-{uuid4().hex}"
            tool_message = {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
                "tool_call_id": tool_call_id,
            }

            # store tool result for future reference
            self._session_store.set_tool_result(session_id, tool_name, tool_result)
            messages.append({"role": "assistant", "content": reply})
            messages.append(tool_message)

            second_reply = await self._generate_response(messages)
            self._session_store.append_message(session_id, "assistant", second_reply)
            return {
                "reply": second_reply,
                "tool_used": tool_name,
                "tool_result": tool_result,
            }

        self._session_store.append_message(session_id, "assistant", reply)
        return {"reply": reply, "tool_used": None}

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
