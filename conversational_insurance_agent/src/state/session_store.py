from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from ..config import get_settings
from .client_context import (
    ClientDatum,
    build_verification_fields,
    deserialize_clients,
    merge_client_records,
    serialize_clients,
)


class ConversationSessionStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._prefix = "conversation"
        self._ttl_seconds = 60 * 60 * 24

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def _write(self, session_id: str, session: Dict[str, Any]) -> None:
        session.setdefault("messages", [])
        session.setdefault("clients", [])
        payload = json.dumps(session)
        self._redis.set(self._key(session_id), payload, ex=self._ttl_seconds)

    def get(self, session_id: str) -> Dict[str, Any]:
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return {"messages": [], "clients": []}
        session = json.loads(raw)
        session.setdefault("messages", [])
        session.setdefault("clients", [])
        return session

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get(session_id)
        session.setdefault("messages", []).append({"role": role, "content": content})
        self._write(session_id, session)

    def set_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        session = self.get(session_id)
        tool_results = session.setdefault("tool_results", {})
        tool_results[tool_name] = result
        self._write(session_id, session)

    def get_tool_result(self, session_id: str, tool_name: str) -> Optional[Any]:
        session = self.get(session_id)
        return session.get("tool_results", {}).get(tool_name)

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def merge_clients(self, session_id: str, clients: List[ClientDatum], source: Optional[str] = None) -> List[ClientDatum]:
        if not clients:
            return self.get_clients(session_id)

        normalized: List[ClientDatum] = []
        for client in clients:
            if source and not client.source:
                copier = getattr(client, "model_copy", None)
                if callable(copier):
                    normalized.append(copier(update={"source": source}))
                else:  # pragma: no cover - pydantic v1 compatibility
                    normalized.append(client.copy(update={"source": source}))  # type: ignore[attr-defined]
            else:
                normalized.append(client)

        session = self.get(session_id)
        existing_clients = deserialize_clients(session.get("clients", []))
        merged_clients = merge_client_records(existing_clients, normalized)
        session["clients"] = serialize_clients(merged_clients)
        self._write(session_id, session)
        return merged_clients

    def get_clients(self, session_id: str) -> List[ClientDatum]:
        session = self.get(session_id)
        raw_clients = session.get("clients", [])
        if not raw_clients:
            return []
        return deserialize_clients(raw_clients)

    def evaluate_payment_readiness(self, session_id: str) -> Dict[str, Any]:
        clients = self.get_clients(session_id)
        if not clients:
            return {
                "status": "missing_clients",
                "message": "I don't have any traveller profile yet.",
            }

        for client in clients:
            missing = client.required_missing_fields()
            if missing:
                return {
                    "status": "missing_fields",
                    "client_id": client.client_id,
                    "missing": missing,
                }

            if client.verification.status != "confirmed":
                verification_fields = client.verification.fields or build_verification_fields(client)
                return {
                    "status": "unverified",
                    "client_id": client.client_id,
                    "fields": verification_fields,
                }

        return {
            "status": "ready",
            "client_id": clients[0].client_id,
        }

    def request_verification(self, session_id: str, client_id: Optional[str], fields: Dict[str, Any]) -> None:
        session = self.get(session_id)
        updated = False
        for client in session.get("clients", []):
            if self._matches_client(client, client_id):
                verification = client.setdefault("verification", {})
                verification["status"] = "pending"
                verification["fields"] = fields
                verification["requested_at"] = _iso_now()
                verification.pop("confirmed_at", None)
                updated = True
        if updated:
            self._write(session_id, session)

    def try_mark_verification(self, session_id: str, user_message: str) -> bool:
        text = user_message.strip().lower()
        if not text:
            return False
        if "?" in text:
            return False

        confirmation_phrases = [
            "confirm",
            "confirmed",
            "looks good",
            "correct",
            "go ahead",
            "approve",
            "proceed",
            "verified",
        ]
        accepted_starts = {"yes", "yup", "yeah", "sure", "ok", "okay"}

        is_confirmation = any(phrase in text for phrase in confirmation_phrases)
        if not is_confirmation:
            first_token = text.split()[0] if text.split() else ""
            if first_token in accepted_starts:
                is_confirmation = True

        if not is_confirmation:
            return False

        session = self.get(session_id)
        updated = False
        for client in session.get("clients", []):
            verification = client.get("verification", {})
            if verification.get("status") == "pending":
                verification["status"] = "confirmed"
                verification["confirmed_at"] = _iso_now()
                verification.setdefault("fields", {})
                updated = True

        if updated:
            self._write(session_id, session)

        return updated

    @staticmethod
    def _matches_client(client_payload: Dict[str, Any], client_id: Optional[str]) -> bool:
        if client_id and client_payload.get("client_id") == client_id:
            return True
        if not client_id:
            return True
        personal_info = client_payload.get("personal_info", {})
        passport = personal_info.get("passport_number") or personal_info.get("passportNumber")
        if passport and passport == client_id:
            return True
        return False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
