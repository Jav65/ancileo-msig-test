from src.core.profile_guidance import compose_profile_guidance
from src.state.client_context import ClientDatum


def test_compose_profile_guidance_returns_none_for_empty_clients() -> None:
    assert compose_profile_guidance([]) is None


def test_compose_profile_guidance_identifies_rich_profile() -> None:
    client = ClientDatum.model_validate(
        {
            "client_id": "cust-alice",
            "source": "partner_portal",
            "personal_info": {
                "name": "Alice Tan",
                "email_address": "alice@example.com",
                "phone_number": "+65 8123 4455",
                "date_of_birth": "1990-06-15",
                "place_of_residence": "Malaysia",
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
                    "metadata": {"activity": "winter sports"},
                }
            ],
            "interests": ["family coverage", "winter sports"],
        }
    )

    guidance = compose_profile_guidance([client])
    assert guidance is not None
    assert guidance.status == "rich"
    summary = guidance.summary_text
    assert "Alice Tan" in summary
    assert "\"missing_fields\": []" in summary
    assert "claims_recommendation" in summary


def test_compose_profile_guidance_identifies_partial_profile() -> None:
    client = ClientDatum.model_validate(
        {
            "client_id": "cust-casey",
            "source": "partner_portal",
            "personal_info": {
                "name": "Casey Lin",
                "email_address": "casey@example.com",
            },
            "trips": [
                {
                    "trip_id": "casey-france-2025",
                    "destination": "France",
                    "start_date": "2025-04-03",
                    "trip_type": "single",
                }
            ],
        }
    )

    guidance = compose_profile_guidance([client])
    assert guidance is not None
    assert guidance.status == "partial"
    summary = guidance.summary_text
    assert "Profile is incomplete" in summary
    assert "Trip details" in summary or "Trip end date" in summary
