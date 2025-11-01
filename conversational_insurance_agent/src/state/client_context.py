from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

try:  # Pydantic v2 support
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - fallback for pydantic v1
    ConfigDict = None  # type: ignore[misc, assignment]


def _personal_field_names(model: type[BaseModel]) -> Iterable[str]:
    fields = getattr(model, "model_fields", None)
    if fields is None:  # pydantic v1
        fields = getattr(model, "__fields__", {})
    return fields.keys()


def _model_dump(model: BaseModel, **kwargs: Any) -> Dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(**kwargs)
    return model.dict(**kwargs)  # type: ignore[return-value]


def _model_copy(model: BaseModel, **kwargs: Any) -> BaseModel:
    copier = getattr(model, "model_copy", None)
    if callable(copier):
        return copier(**kwargs)
    return model.copy(**kwargs)  # type: ignore[return-value]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return True
    return False


class PersonalInfo(BaseModel):
    name: Optional[str] = None
    email_address: Optional[str] = Field(None, alias="emailAddress")
    phone_number: Optional[str] = Field(None, alias="phoneNumber")
    date_of_birth: Optional[date] = Field(None, alias="dateOfBirth")
    place_of_residence: Optional[str] = Field(None, alias="placeOfResidence")
    passport_number: Optional[str] = Field(None, alias="passportNumber")

    if ConfigDict:  # pragma: no branch - static configuration
        model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    else:  # pragma: no cover - compatibility shim for pydantic v1
        class Config:
            allow_population_by_field_name = True
            extra = "allow"


class TripDetails(BaseModel):
    trip_id: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[date] = Field(None, alias="startDate")
    end_date: Optional[date] = Field(None, alias="endDate")
    trip_type: Optional[Literal["single", "round"]] = Field(None, alias="tripType")
    trip_cost: Optional[float] = Field(None, alias="tripCost")
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    if ConfigDict:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True, extra="allow")

    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            allow_population_by_field_name = True
            extra = "allow"

    def missing_fields(self) -> List[str]:
        missing: List[str] = []
        if _is_blank(self.destination):
            missing.append("Trip destination")
        if _is_blank(self.start_date):
            missing.append("Trip start date")
        if _is_blank(self.end_date):
            missing.append("Trip end date")
        if _is_blank(self.trip_type):
            missing.append("Trip type")
        if _is_blank(self.trip_cost):
            missing.append("Trip cost")
        return missing


class VerificationRecord(BaseModel):
    status: Literal["unknown", "pending", "confirmed"] = "unknown"
    requested_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict)

    if ConfigDict:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True, extra="allow")

    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            allow_population_by_field_name = True
            extra = "allow"


class ClientDatum(BaseModel):
    client_id: Optional[str] = None
    source: Optional[str] = None
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    trips: List[TripDetails] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    verification: VerificationRecord = Field(default_factory=VerificationRecord)

    if ConfigDict:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True, extra="allow")

    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            allow_population_by_field_name = True
            extra = "allow"

    def required_missing_fields(self) -> List[str]:
        missing: List[str] = []
        required_personal: List[Tuple[str, str]] = [
            ("name", "Name"),
            ("email_address", "Email address"),
            ("phone_number", "Phone number"),
            ("date_of_birth", "Date of birth"),
            ("place_of_residence", "Place of residence"),
            ("passport_number", "Passport number"),
        ]
        for field_name, label in required_personal:
            if _is_blank(getattr(self.personal_info, field_name, None)):
                missing.append(label)

        if not self.trips:
            missing.append("Trip details")
            return missing

        primary_trip = select_preferred_trip(self)
        if primary_trip is None:
            missing.append("Trip details")
        else:
            missing.extend(primary_trip.missing_fields())

        return missing


def select_preferred_trip(client: ClientDatum) -> Optional[TripDetails]:
    if not client.trips:
        return None
    for trip in client.trips:
        if not trip.missing_fields():
            return trip
    return client.trips[0]


def merge_client_records(
    existing_clients: List[ClientDatum], incoming_clients: List[ClientDatum]
) -> List[ClientDatum]:
    merged: List[ClientDatum] = [client for client in existing_clients]
    for candidate in incoming_clients:
        match = _find_matching_client(merged, candidate)
        if match is None:
            merged.append(candidate)
            continue
        _merge_client(match, candidate)
    return merged


def serialize_clients(clients: List[ClientDatum]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for client in clients:
        raw = _model_dump(client, exclude_none=True, by_alias=True)
        serialized.append(_to_jsonable(raw))
    return serialized


def deserialize_clients(payload: Iterable[Dict[str, Any]]) -> List[ClientDatum]:
    result: List[ClientDatum] = []
    for data in payload:
        result.append(ClientDatum.model_validate(data))
    return result


def build_verification_fields(client: ClientDatum) -> Dict[str, Any]:
    trip = select_preferred_trip(client)
    fields: Dict[str, Any] = {
        "name": client.personal_info.name,
        "email_address": client.personal_info.email_address,
        "passport_number": client.personal_info.passport_number,
        "phone_number": client.personal_info.phone_number,
    }
    if trip is not None:
        fields.update(
            {
                "destination": trip.destination,
                "trip_type": trip.trip_type,
                "trip_cost": trip.trip_cost,
                "travel_dates": _format_trip_dates(trip),
            }
        )
    return {k: v for k, v in fields.items() if not _is_blank(v)}


def _format_trip_dates(trip: TripDetails) -> Optional[str]:
    if trip.start_date and trip.end_date:
        return f"{trip.start_date} -> {trip.end_date}"
    if trip.start_date:
        return str(trip.start_date)
    if trip.end_date:
        return str(trip.end_date)
    return None


def _find_matching_client(
    existing_clients: List[ClientDatum], candidate: ClientDatum
) -> Optional[ClientDatum]:
    candidate_keys = _client_identity_keys(candidate)
    if not candidate_keys:
        # Attempt relaxed matching when no strong identifiers are available.
        if len(existing_clients) == 1:
            return existing_clients[0]

        if candidate.source:
            source_matches = [client for client in existing_clients if client.source == candidate.source]
            if len(source_matches) == 1:
                return source_matches[0]

        candidate_name = (candidate.personal_info.name or "").strip().lower()
        if candidate_name:
            name_matches = [
                client
                for client in existing_clients
                if (client.personal_info.name or "").strip().lower() == candidate_name
            ]
            if len(name_matches) == 1:
                return name_matches[0]
            
        return None

    for existing in existing_clients:
        existing_keys = _client_identity_keys(existing)
        if existing_keys.intersection(candidate_keys):
            return existing
    return None


def _client_identity_keys(client: ClientDatum) -> set[Tuple[str, str]]:
    keys: set[Tuple[str, str]] = set()
    if not _is_blank(client.client_id):
        keys.add(("client_id", str(client.client_id).lower()))

    passport = client.personal_info.passport_number
    if not _is_blank(passport):
        keys.add(("passport_number", passport.strip().upper()))

    email = client.personal_info.email_address
    if not _is_blank(email):
        keys.add(("email_address", email.strip().lower()))

    phone = client.personal_info.phone_number
    if not _is_blank(phone):
        phone_normalized = "".join(ch for ch in phone if ch.isdigit())
        keys.add(("phone_number", phone_normalized))

    return keys


def _merge_client(target: ClientDatum, source: ClientDatum) -> None:
    prefer_source = source.verification.status == "confirmed"

    if _is_blank(target.client_id) and not _is_blank(source.client_id):
        target.client_id = source.client_id
    if _is_blank(target.source) and not _is_blank(source.source):
        target.source = source.source

    target.personal_info = _merge_personal_info(target.personal_info, source.personal_info, prefer_source)

    target.interests = _merge_interests(target.interests, source.interests)

    target.trips = _merge_trips(target.trips, source.trips, prefer_source)

    if source.extra:
        target.extra = {**target.extra, **source.extra}

    target.verification = _merge_verification(target.verification, source.verification)


def _merge_personal_info(
    target: PersonalInfo, source: PersonalInfo, prefer_source: bool
) -> PersonalInfo:
    updates: Dict[str, Any] = {}
    for field_name in _personal_field_names(PersonalInfo):
        new_value = getattr(source, field_name, None)
        if _is_blank(new_value):
            continue
        current_value = getattr(target, field_name, None)
        if _is_blank(current_value) or prefer_source:
            updates[field_name] = new_value
    if not updates:
        return target
    return _model_copy(target, update=updates)


def _merge_interests(existing: List[str], incoming: List[str]) -> List[str]:
    combined: List[str] = []
    seen: set[str] = set()
    for value in [*existing, *incoming]:
        if _is_blank(value):
            continue
        normalized = value.strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(normalized)
    return combined


def _merge_trips(
    existing_trips: List[TripDetails], incoming_trips: List[TripDetails], prefer_source: bool
) -> List[TripDetails]:
    merged: List[TripDetails] = [trip for trip in existing_trips]
    for trip in incoming_trips:
        match_index = _locate_trip_index(merged, trip)
        if match_index is None:
            merged.append(trip)
            continue
        merged[match_index] = _merge_trip(merged[match_index], trip, prefer_source)
    return merged


def _locate_trip_index(trips: List[TripDetails], candidate: TripDetails) -> Optional[int]:
    candidate_key = _trip_identity_key(candidate)
    if candidate_key is None:
        return None
    for index, trip in enumerate(trips):
        if _trip_identity_key(trip) == candidate_key:
            return index
    return None


def _merge_trip(
    base_trip: TripDetails, new_trip: TripDetails, prefer_source: bool
) -> TripDetails:
    updates: Dict[str, Any] = {}
    for field_name in _personal_field_names(TripDetails):
        new_value = getattr(new_trip, field_name, None)
        if _is_blank(new_value):
            continue
        current_value = getattr(base_trip, field_name, None)
        if field_name == "metadata":
            updates[field_name] = {**(current_value or {}), **(new_value or {})}
            continue
        if _is_blank(current_value) or prefer_source:
            updates[field_name] = new_value
    if not updates:
        return base_trip
    return _model_copy(base_trip, update=updates)


def _trip_identity_key(trip: TripDetails) -> Optional[Tuple[str, str, str, str]]:
    if not _is_blank(trip.trip_id):
        return ("trip_id", str(trip.trip_id), "", "")
    if not _is_blank(trip.destination) and not _is_blank(trip.start_date):
        start = str(trip.start_date)
        end = str(trip.end_date) if trip.end_date else ""
        trip_type = trip.trip_type or ""
        destination = trip.destination.strip().lower()  # type: ignore[union-attr]
        return (destination, start, end, trip_type)
    return None


def _merge_verification(
    current: VerificationRecord, incoming: VerificationRecord
) -> VerificationRecord:
    priority = {"unknown": 0, "pending": 1, "confirmed": 2}
    current_priority = priority.get(current.status, 0)
    incoming_priority = priority.get(incoming.status, 0)

    if incoming_priority > current_priority:
        updates: Dict[str, Any] = {
            "status": incoming.status,
            "fields": incoming.fields or current.fields,
        }
        if incoming.status == "pending" and _is_blank(incoming.requested_at):
            updates["requested_at"] = _iso_now()
        else:
            updates["requested_at"] = incoming.requested_at or current.requested_at
        updates["confirmed_at"] = incoming.confirmed_at or (
            _iso_now() if incoming.status == "confirmed" else current.confirmed_at
        )
        return _model_copy(current, update=updates)

    if incoming_priority == current_priority:
        merged_fields = {**current.fields, **incoming.fields}
        updates = {"fields": merged_fields}
        if incoming.requested_at and (not current.requested_at or incoming.requested_at > current.requested_at):
            updates["requested_at"] = incoming.requested_at
        if incoming.confirmed_at and (not current.confirmed_at or incoming.confirmed_at > current.confirmed_at):
            updates["confirmed_at"] = incoming.confirmed_at
        return _model_copy(current, update=updates)

    if current_priority == 0 and incoming_priority == 0 and incoming.fields:
        return _model_copy(current, update={"fields": incoming.fields})

    return current


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value

