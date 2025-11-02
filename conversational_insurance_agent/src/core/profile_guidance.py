from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from ..state.client_context import ClientDatum, TripDetails, select_preferred_trip


@dataclass
class ProfileGuidance:
    status: str
    summary_text: str


def compose_profile_guidance(clients: List[ClientDatum]) -> Optional[ProfileGuidance]:
    if not clients:
        return None

    entries: List[Dict[str, Any]] = []
    complete = 0
    partial = 0

    for index, client in enumerate(clients, start=1):
        label = client.personal_info.name or client.client_id or f"Client {index}"
        missing = client.required_missing_fields()
        if missing:
            partial += 1
        else:
            complete += 1

        trip = select_preferred_trip(client)
        entry: Dict[str, Any] = {
            "label": label,
            "client_id": client.client_id,
            "source": client.source,
            "verification": client.verification.status,
            "missing_fields": missing,
            "personal_info": _compact_personal_info(client),
            "interests": client.interests or None,
        }

        if trip:
            entry["trip"] = _serialize_trip(trip)
            if not missing:
                entry["tool_inputs"] = _build_tool_hints(client, trip)

        entries.append({k: v for k, v in entry.items() if v not in (None, [], {})})

    status = "rich" if complete else ("partial" if partial else "sparse")

    instructions = _build_instructions(status)
    payload = {
        "status": status,
        "clients": entries,
        "workflow": instructions,
    }

    summary = "[Integrated Traveller Data]\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return ProfileGuidance(status=status, summary_text=summary)


def _compact_personal_info(client: ClientDatum) -> Dict[str, Any]:
    info = client.personal_info
    payload = {
        "name": info.name,
        "email": info.email_address,
        "phone": info.phone_number,
        "residence": info.place_of_residence,
        "passport": info.passport_number,
        "date_of_birth": _format_date(info.date_of_birth),
    }
    return {k: v for k, v in payload.items() if v}


def _serialize_trip(trip: TripDetails) -> Dict[str, Any]:
    payload = {
        "trip_id": trip.trip_id,
        "destination": trip.destination,
        "start_date": _format_date(trip.start_date),
        "end_date": _format_date(trip.end_date),
        "trip_type": trip.trip_type,
        "trip_cost": trip.trip_cost,
    }
    if trip.metadata:
        payload["metadata"] = trip.metadata
    if trip.notes:
        payload["notes"] = trip.notes
    return {k: v for k, v in payload.items() if v is not None and v != ""}


def _build_tool_hints(client: ClientDatum, trip: TripDetails) -> Dict[str, Any]:
    metadata = trip.metadata or {}
    activity = metadata.get("activity") or (client.interests[0] if client.interests else None)
    tool_inputs: Dict[str, Any] = {
        "claims_recommendation": {
            "destination": trip.destination,
            "activity": activity,
            "trip_cost": trip.trip_cost,
        }
    }

    return {name: {k: v for k, v in params.items() if v is not None} for name, params in tool_inputs.items()}


def _build_instructions(status: str) -> List[str]:
    instructions = [
        "Surface the integration data, confirm accuracy with the traveller, and note any missing items.",
        "Always keep responses concise, empathetic, and cite policy sources in answers.",
        "Never initiate payment until all required fields are present and the traveller has explicitly confirmed the profile.",
    ]

    if status == "rich":
        instructions.insert(
            0,
            "Profile is complete. After confirmation, immediately run `claims_recommendation` and follow up with `policy_research` to produce tailored options.",
        )
    else:
        instructions.insert(
            0,
            "Profile is incomplete. Ask targeted questions to capture the missing information before running recommendation tools.",
        )

    return instructions


def _format_date(value: Optional[date]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()
