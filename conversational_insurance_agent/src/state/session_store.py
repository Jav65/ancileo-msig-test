from __future__ import annotations

import json
from typing import Any, Dict, Optional

import redis

from ..config import get_settings


class ConversationSessionStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._prefix = "conversation"
        self._ttl_seconds = 60 * 60 * 24

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def get(self, session_id: str) -> Dict[str, Any]:
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return {"messages": []}
        return json.loads(raw)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get(session_id)
        session.setdefault("messages", []).append({"role": role, "content": content})
        self._redis.set(self._key(session_id), json.dumps(session), ex=self._ttl_seconds)

    def set_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        session = self.get(session_id)
        tool_results = session.setdefault("tool_results", {})
        tool_results[tool_name] = result
        self._redis.set(self._key(session_id), json.dumps(session), ex=self._ttl_seconds)

    def get_tool_result(self, session_id: str, tool_name: str) -> Optional[Any]:
        session = self.get(session_id)
        return session.get("tool_results", {}).get(tool_name)

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))
