from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from email.utils import parsedate_to_datetime

from ..state.client_context import ClientDatum, TripDetails
from ..utils.logging import logger


GMAIL_TICKET_QUERY = (
    "subject:(itinerary OR booking OR reservation) (flight OR airline OR ticket) newer_than:365d"
)

DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)

CURRENCY_PATTERN = re.compile(
    r"(?P<currency>SGD|USD|EUR|GBP|AUD|CAD|JPY|MYR|THB|IDR|INR|PHP)\s?(?P<amount>[\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)
DESTINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Destination\s*[:\-]\s*(?P<value>.+)", re.IGNORECASE),
    re.compile(r"To\s*[:\-]\s*(?P<value>.+)", re.IGNORECASE),
    re.compile(r"Arriving\s+in\s+(?P<value>[A-Za-z\s]+)", re.IGNORECASE),
)
FALLBACK_DESTINATION_PATTERN = re.compile(
    r"(?:flight|trip|travel|journey|itinerary)\s+(?:to|for)\s+(?P<value>[A-Za-z\s]{3,})",
    re.IGNORECASE,
)
DATE_TOKEN_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b"
)
DATE_CONTEXT_WINDOW = 35
TRIP_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("one-way", "single"),
    ("one way", "single"),
    ("round trip", "round"),
    ("return", "round"),
)


@dataclass
class GmailTripCandidate:
    trip: TripDetails
    message_url: str


class GmailDataError(RuntimeError):
    """Raised when Gmail data cannot be fetched or parsed."""


def fetch_travel_client(
    credentials: Credentials,
    *,
    profile: Optional[Dict[str, Any]] = None,
    max_messages: int = 10,
) -> ClientDatum:
    """Fetch recent travel-related emails and build a client profile."""

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    except Exception as exc:  # pragma: no cover - googleapiclient discovery errors
        raise GmailDataError("Unable to initialise Gmail service") from exc

    user_profile: Dict[str, Any]
    try:
        user_profile = service.users().getProfile(userId="me").execute()  # type: ignore[no-untyped-call]
    except HttpError as exc:  # pragma: no cover - requires Gmail connectivity
        logger.exception("gmail.profile.fetch_failed", error=str(exc))
        raise GmailDataError("Unable to fetch Gmail profile") from exc

    email_address = user_profile.get("emailAddress") or (profile or {}).get("email")
    if not email_address:
        raise GmailDataError("Unable to determine authenticated email address")

    personal_info: Dict[str, Any] = {"email_address": email_address}
    display_name = (profile or {}).get("name") or (profile or {}).get("given_name")
    if display_name:
        personal_info["name"] = display_name

    client = ClientDatum(
        client_id=email_address,
        source="gmail_portal",
        personal_info=personal_info,
        trips=[],
        interests=["travel", "airfare"],
        extra={
            "gmail": {
                "historyId": user_profile.get("historyId"),
                "messagesTotal": user_profile.get("messagesTotal"),
            }
        },
    )

    try:
        response = (
            service.users()
            .messages()
            .list(userId="me", q=GMAIL_TICKET_QUERY, maxResults=max_messages)
            .execute()
        )
    except HttpError as exc:  # pragma: no cover - requires Gmail connectivity
        logger.exception("gmail.messages.fetch_failed", error=str(exc))
        raise GmailDataError("Unable to fetch Gmail messages") from exc

    message_ids: Iterable[Dict[str, Any]] = response.get("messages", []) if response else []

    trips: List[TripDetails] = []
    for message_meta in message_ids:
        message_id = message_meta.get("id")
        if not message_id:
            continue
        try:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:  # pragma: no cover - requires Gmail connectivity
            logger.warning("gmail.message.fetch_failed", message_id=message_id, error=str(exc))
            continue

        candidate = _parse_trip_candidate(message)
        if candidate is None:
            continue
        trips.append(candidate.trip)

    if trips:
        client.trips = trips

    missing_fields = client.required_missing_fields()
    if missing_fields:
        client.verification.fields.update({"missing": missing_fields})

    return client


def _parse_trip_candidate(message: Dict[str, Any]) -> Optional[GmailTripCandidate]:
    payload = message.get("payload", {})
    headers = {header.get("name", "").lower(): header.get("value", "") for header in payload.get("headers", [])}
    subject = headers.get("subject", "")
    date_header = headers.get("date", "")
    sent_at = _parse_datetime(date_header)
    body_text = _extract_body_text(payload) or message.get("snippet", "")

    combined_text = f"{subject}\n{body_text}"

    destination = _extract_destination(combined_text)
    start_date, end_date = _extract_dates(combined_text)
    trip_type = _infer_trip_type(combined_text, start_date, end_date)
    trip_cost = _extract_cost(combined_text)

    message_url = f"https://mail.google.com/mail/u/0/#inbox/{message.get('id')}" if message.get("id") else ""

    metadata: Dict[str, Any] = {
        "subject": subject,
        "snippet": message.get("snippet"),
        "sentAt": sent_at.isoformat() if sent_at else None,
        "gmailMessageId": message.get("id"),
        "threadId": message.get("threadId"),
        "headers": _safe_headers(payload.get("headers", [])),
        "messageUrl": message_url or None,
    }

    if not any([destination, start_date, end_date, trip_cost]):
        return None

    trip = TripDetails(
        trip_id=message.get("id"),
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        trip_type=trip_type,
        trip_cost=trip_cost,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    return GmailTripCandidate(trip=trip, message_url=message_url)


def _extract_body_text(payload: Dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")

    if data:
        decoded = _decode_body(data)
        if mime_type == "text/html":
            return _html_to_text(decoded)
        return decoded

    parts = payload.get("parts", [])
    for part in parts:
        part_type = part.get("mimeType")
        if part_type in {"text/plain", "text/html"}:
            text = _extract_body_text(part)
            if text:
                return text
    return ""


def _decode_body(data: str) -> str:
    try:
        decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
        return decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover - defensive
        return ""


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_destination(text: str) -> Optional[str]:
    for pattern in DESTINATION_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group("value").strip()
            if value:
                return value

    # Fallback heuristic based on subject structure e.g. "Flight to Tokyo"
    subject_match = FALLBACK_DESTINATION_PATTERN.search(text)
    if subject_match:
        return subject_match.group(1).strip()
    return None


def _extract_dates(text: str) -> tuple[Optional[date], Optional[date]]:
    start: Optional[date] = None
    end: Optional[date] = None

    for match in DATE_TOKEN_PATTERN.finditer(text):
        token = match.group(1)
        parsed = _parse_date(token)
        if not parsed:
            continue

        context_start = max(0, match.start() - DATE_CONTEXT_WINDOW)
        context = text[context_start:match.start()].lower()
        if any(keyword in context for keyword in ("return", "arrive", "back", "inbound", "home")):
            if end is None or parsed > end:
                end = parsed
            continue

        if any(keyword in context for keyword in ("depart", "outbound", "leave", "start", "fly out")):
            if start is None or parsed < start:
                start = parsed
            continue

        if start is None:
            start = parsed
        elif end is None:
            end = parsed

    if start and not end:
        end = start

    return start, end


def _parse_date(value: str) -> Optional[date]:
    cleaned = value.strip()
    cleaned = re.sub(r"(st|nd|rd|th)", "", cleaned)
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            # Two-digit year handling
            if dt.year < 100:
                dt = dt.replace(year=2000 + dt.year)
            return dt.date()
        except ValueError:
            continue
    return None


def _infer_trip_type(text: str, start: Optional[date], end: Optional[date]) -> Optional[str]:
    lowered = text.lower()
    for marker, trip_type in TRIP_TYPE_HINTS:
        if marker in lowered:
            return trip_type

    if start and end and start != end:
        return "round"
    if start and end and start == end:
        return "single"
    return None


def _extract_cost(text: str) -> Optional[float]:
    match = CURRENCY_PATTERN.search(text)
    if not match:
        return None
    amount_str = match.group("amount").replace(",", "")
    try:
        return float(amount_str)
    except ValueError:
        return None


def _safe_headers(headers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe: List[Dict[str, Any]] = []
    for header in headers:
        name = header.get("name")
        value = header.get("value")
        if not name:
            continue
        safe.append({"name": name, "value": value})
    return safe


def _parse_datetime(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
