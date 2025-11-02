from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..state.client_context import ClientDatum


@dataclass(frozen=True)
class MockUserRecord:
    username: str
    password: str
    display_name: str
    client_payload: Dict[str, Any]
    past_policies: List[Dict[str, Any]]

    def build_client(self) -> ClientDatum:
        return ClientDatum.model_validate(self.client_payload)

    @property
    def session_id(self) -> str:
        client_id = self.client_payload.get("client_id") or self.username
        return f"integration-{client_id}"


def _load_users() -> Dict[str, MockUserRecord]:
    return {
        "alice@example.com": MockUserRecord(
            username="alice@example.com",
            password="travel123",
            display_name="Alice Tan",
            client_payload={
                "client_id": "cust-alice",
                "source": "partner_portal",
                "personal_info": {
                    "name": "Alice Tan",
                    "email_address": "alice@example.com",
                    "phone_number": "+65 8123 4455",
                    "date_of_birth": "1990-06-15",
                    "place_of_residence": "Singapore",
                    "passport_number": "E1234567K",
                },
                "trips": [
                    {
                        "trip_id": "alice-japan-2025",
                        "destination": "Hokkaido, Japan",
                        "start_date": "2025-12-10",
                        "end_date": "2025-12-20",
                        "trip_type": "round",
                        "trip_cost": 4800.0,
                        "notes": "Ski holiday with family",
                        "metadata": {
                            "activity": "winter sports",
                            "travellers": 3,
                        },
                    }
                ],
                "interests": ["family coverage", "winter sports"],
                "extra": {
                    "preferred_currency": "SGD",
                    "loyalty_tier": "gold",
                },
            },
            past_policies=[
                {
                    "policy": "TravelEasy Premier",
                    "purchase_date": "2024-01-05",
                    "coverage": "Worldwide Annual Multi-Trip",
                    "claims": "No claims filed",
                },
                {
                    "policy": "Scootsurance Flexi",
                    "purchase_date": "2023-06-12",
                    "coverage": "Single Trip to Australia",
                    "claims": "Delayed baggage claim settled",
                },
            ],
        ),
        "casey@example.com": MockUserRecord(
            username="casey@example.com",
            password="explore",
            display_name="Casey Lin",
            client_payload={
                "client_id": "cust-casey",
                "source": "partner_portal",
                "personal_info": {
                    "name": "Casey Lin",
                    "email_address": "casey@example.com",
                    "place_of_residence": "Singapore",
                },
                "trips": [
                    {
                        "trip_id": "casey-europe-2025",
                        "destination": "Western Europe",
                        "start_date": "2025-04-03",
                        "trip_type": "single",
                        "metadata": {
                            "activity": "city hopping",
                        },
                    }
                ],
                "interests": ["adventure activities"],
                "extra": {
                    "preferred_currency": "EUR",
                },
            },
            past_policies=[
                {
                    "policy": "TravelEasy Classic",
                    "purchase_date": "2022-11-22",
                    "coverage": "Single Trip to Thailand",
                    "claims": "Medical expense reimbursement",
                }
            ],
        ),
        "ben@travelco.sg": MockUserRecord(
            username="ben@travelco.sg",
            password="partner",
            display_name="Benjamin Koh",
            client_payload={
                "client_id": "cust-ben",
                "source": "travelco_portal",
                "personal_info": {
                    "name": "Benjamin Koh",
                    "email_address": "ben@travelco.sg",
                    "phone_number": "+65 8899 2211",
                    "passport_number": "S9988776P",
                    "place_of_residence": "Singapore",
                },
                "trips": [],
                "interests": ["golf", "luxury hotels"],
                "extra": {
                    "corporate_account": "TravelCo-SG-881",
                },
            },
            past_policies=[
                {
                    "policy": "Corporate Shield",
                    "purchase_date": "2023-08-19",
                    "coverage": "Regional Multi-Trip",
                    "claims": "Lost baggage compensation",
                }
            ],
        ),
    }


_USERS = _load_users()


def authenticate_user(username: str, password: str) -> Optional[MockUserRecord]:
    record = _USERS.get(username.lower())
    if record and record.password == password:
        return record
    return None


def get_user(username: str) -> Optional[MockUserRecord]:
    return _USERS.get(username.lower())


def list_users() -> Iterable[MockUserRecord]:
    return _USERS.values()
