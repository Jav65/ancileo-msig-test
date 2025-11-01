import unittest
from datetime import date

from src.state.client_context import ClientDatum, PersonalInfo
from src.state.session_store import _enrich_client_from_payment_payload


class PaymentContextEnrichmentTest(unittest.TestCase):
    def test_enrich_client_with_complete_metadata(self) -> None:
        client = ClientDatum(
            personal_info=PersonalInfo(
                name="Javier Wong",
                phone_number="+6591234567",
            ),
            trips=[],
        )

        payload = {
            "customer_email": "j4vierwong0605@gmail.com",
            "metadata": {
                "passport_number": "E1281932",
                "date_of_birth": "06 April 1999",
                "place_of_residence": "Singapore",
                "trip_destination": "Osaka",
                "trip_start_date": "2025-11-03",
                "trip_end_date": "2025-11-10",
                "trip_type": "Single",
                "trip_cost": "500.00",
            },
        }

        updated = _enrich_client_from_payment_payload(client, payload)

        self.assertTrue(updated)
        self.assertEqual(client.personal_info.email_address, "j4vierwong0605@gmail.com")
        self.assertEqual(client.personal_info.passport_number, "E1281932")
        self.assertEqual(client.personal_info.place_of_residence, "Singapore")
        self.assertEqual(client.personal_info.date_of_birth, date(1999, 4, 6))

        self.assertEqual(len(client.trips), 1)
        trip = client.trips[0]
        self.assertEqual(trip.destination, "Osaka")
        self.assertEqual(trip.start_date, date(2025, 11, 3))
        self.assertEqual(trip.end_date, date(2025, 11, 10))
        self.assertEqual(trip.trip_type, "single")
        self.assertEqual(trip.trip_cost, 500.0)

        self.assertEqual(client.required_missing_fields(), [])

    def test_enrich_client_with_irrelevant_metadata_returns_false(self) -> None:
        client = ClientDatum(personal_info=PersonalInfo(name="Aisha"), trips=[])

        payload = {
            "metadata": {
                "notes": "no structured traveller data",
                "reference_id": "REF-12345",
            }
        }

        updated = _enrich_client_from_payment_payload(client, payload)

        self.assertFalse(updated)
        self.assertIn("Email address", client.required_missing_fields())


if __name__ == "__main__":
    unittest.main()
