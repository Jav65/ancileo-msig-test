from __future__ import annotations

import base64
from datetime import date

import pytest

from src.services.gmail_ticket_scraper import _extract_dates, _extract_destination, _parse_trip_candidate


def _encode_body(content: str) -> str:
    return base64.urlsafe_b64encode(content.encode("utf-8")).decode("ascii")


def _build_message(body: str, *, subject: str = "Flight to Tokyo", message_id: str = "msg-123") -> dict:
    return {
        "id": message_id,
        "threadId": "thr-001",
        "snippet": body[:120],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Fri, 10 Jan 2025 12:45:00 +0000"},
            ],
            "body": {"data": _encode_body(body)},
        },
    }


def test_parse_trip_candidate_extracts_destination_dates_and_cost():
    body = (
        "Booking Reference: SQ12345\n"
        "Destination: Sapporo, Japan\n"
        "Departure: 2025-12-10\n"
        "Return: 2025-12-20\n"
        "Fare: SGD 4,820.50"
    )

    message = _build_message(body)
    candidate = _parse_trip_candidate(message)

    assert candidate is not None, "Expected a trip candidate to be parsed"
    trip = candidate.trip
    assert trip.destination == "Sapporo, Japan"
    assert trip.start_date == date(2025, 12, 10)
    assert trip.end_date == date(2025, 12, 20)
    assert trip.trip_cost == pytest.approx(4820.50)
    assert trip.trip_type == "round"
    assert trip.metadata["messageUrl"]


def test_parse_trip_candidate_returns_none_when_no_travel_signals():
    body = "Thank you for subscribing to our newsletter."
    message = _build_message(body, subject="Monthly Digest")

    assert _parse_trip_candidate(message) is None


def test_extract_helpers_handle_mixed_formats():
    text = (
        "Your outbound flight departs 02/03/2025 from Singapore. "
        "Return 12/03/2025 to Singapore. "
        "Arriving in Paris."
    )

    dest = _extract_destination(text)
    start, end = _extract_dates(text)

    assert dest == "Paris"
    assert start == date(2025, 3, 2)
    assert end == date(2025, 3, 12)
