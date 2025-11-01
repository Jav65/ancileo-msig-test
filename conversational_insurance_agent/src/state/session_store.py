from __future__ import annotations

import json
from datetime import date, datetime, timezone
import re
from typing import Any, Dict, List, Optional

import redis

from ..config import get_settings
from .client_context import (
    ClientDatum,
    TripDetails,
    build_verification_fields,
    deserialize_clients,
    merge_client_records,
    select_preferred_trip,
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

    def apply_payment_context(self, session_id: str, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or not payload:
            return

        session = self.get(session_id)
        raw_clients = session.get("clients", [])
        if not raw_clients:
            return

        clients = deserialize_clients(raw_clients)
        updated = False

        for client in clients:
            if not client.required_missing_fields():
                continue
            if _enrich_client_from_payment_payload(client, payload):
                updated = True

        if not updated:
            return

        session["clients"] = serialize_clients(clients)
        self._write(session_id, session)

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


_PERSONAL_INFO_FIELD_MAP: Dict[str, str] = {
    "customer_email": "email_address",
    "email": "email_address",
    "email_address": "email_address",
    "contact_email": "email_address",
    "name": "name",
    "full_name": "name",
    "customer_name": "name",
    "traveller_name": "name",
    "traveler_name": "name",
    "phone": "phone_number",
    "phone_number": "phone_number",
    "contact_number": "phone_number",
    "mobile": "phone_number",
    "customer_phone": "phone_number",
    "customer_phone_number": "phone_number",
    "date_of_birth": "date_of_birth",
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
    "passport": "passport_number",
    "passport_number": "passport_number",
    "place_of_residence": "place_of_residence",
    "residence": "place_of_residence",
    "home_city": "place_of_residence",
    "city": "place_of_residence",
    "address": "place_of_residence",
}

_TRIP_FIELD_MAP: Dict[str, str] = {
    "destination": "destination",
    "trip_destination": "destination",
    "travel_destination": "destination",
    "destination_city": "destination",
    "start_date": "start_date",
    "trip_start_date": "start_date",
    "departure_date": "start_date",
    "travel_start": "start_date",
    "end_date": "end_date",
    "trip_end_date": "end_date",
    "return_date": "end_date",
    "travel_end": "end_date",
    "trip_type": "trip_type",
    "trip_category": "trip_type",
    "trip_cost": "trip_cost",
    "total_cost": "trip_cost",
    "coverage_amount": "trip_cost",
    "premium_amount": "trip_cost",
}

_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B, %Y",
    "%d %b, %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y.%m.%d",
    "%d.%m.%Y",
]


def _enrich_client_from_payment_payload(client: ClientDatum, payload: Dict[str, Any]) -> bool:
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        metadata = {}

    aggregated: Dict[str, Any] = {}

    def _collect(key: Any, value: Any) -> None:
        normalized_key = _normalize_key(key)
        if not normalized_key:
            return
        if _is_blank_value(value):
            return
        if normalized_key not in aggregated:
            aggregated[normalized_key] = value

    for source_key in (
        "customer_email",
        "customer_name",
        "customer_phone",
        "customer_phone_number",
        "traveller_name",
        "traveler_name",
        "phone_number",
        "passport_number",
        "date_of_birth",
        "place_of_residence",
        "destination",
        "trip_destination",
        "trip_start_date",
        "trip_end_date",
        "trip_type",
        "trip_cost",
    ):
        if source_key in payload:
            _collect(source_key, payload[source_key])

    for key, value in metadata.items():
        _collect(key, value)

    if not aggregated:
        return False

    updated = False

    for key, value in aggregated.items():
        field_name = _PERSONAL_INFO_FIELD_MAP.get(key)
        if not field_name:
            continue
        normalized_value = _normalize_personal_info_value(field_name, value)
        if normalized_value is None:
            continue
        current_value = getattr(client.personal_info, field_name, None)
        if _values_equal(current_value, normalized_value):
            continue
        setattr(client.personal_info, field_name, normalized_value)
        updated = True

    trip_updates: Dict[str, Any] = {}
    for key, value in aggregated.items():
        trip_field = _TRIP_FIELD_MAP.get(key)
        if not trip_field:
            continue
        normalized_value = _normalize_trip_value(trip_field, value)
        if normalized_value is None:
            continue
        trip_updates[trip_field] = normalized_value

    if trip_updates:
        trip = select_preferred_trip(client)
        if trip is None:
            trip = TripDetails()
            client.trips.append(trip)
        for field_name, new_value in trip_updates.items():
            current_value = getattr(trip, field_name, None)
            if _values_equal(current_value, new_value):
                continue
            setattr(trip, field_name, new_value)
            updated = True

    if updated and client.verification.status != "confirmed":
        client.verification.fields = build_verification_fields(client)

    return updated


def _normalize_key(key: Any) -> str:
    if not isinstance(key, str):
        return ""
    trimmed = key.strip()
    if not trimmed:
        return ""
    camel_sanitised = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", trimmed)
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", camel_sanitised)
    return collapsed.lower().strip("_")


def _normalize_personal_info_value(field_name: str, value: Any) -> Optional[Any]:
    if _is_blank_value(value):
        return None
    if field_name == "date_of_birth":
        return _parse_date(value)
    if field_name == "email_address":
        text = str(value).strip()
        return text.lower() if text else None
    if field_name in {"name", "place_of_residence", "passport_number"}:
        text = str(value).strip()
        return text or None
    if field_name == "phone_number":
        text = str(value).strip()
        return text or None
    return None


def _normalize_trip_value(field_name: str, value: Any) -> Optional[Any]:
    if _is_blank_value(value):
        return None
    if field_name in {"start_date", "end_date"}:
        return _parse_date(value)
    if field_name == "trip_cost":
        return _parse_float(value)
    if field_name == "trip_type":
        return _normalize_trip_type(value)
    if field_name == "destination":
        text = str(value).strip()
        return text or None
    return None


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
    if parsed is not None:
        return parsed.date()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_trip_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"single", "single_trip", "single-trip", "one_way", "one-way"}:
        return "single"
    if text in {"round", "round_trip", "round-trip", "return", "return_trip", "roundtrip"}:
        return "round"
    return None


def _values_equal(current: Any, new: Any) -> bool:
    if isinstance(current, str) and isinstance(new, str):
        return current.strip() == new.strip()
    return current == new


def _is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
