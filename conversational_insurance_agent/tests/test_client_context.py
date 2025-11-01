from datetime import date

import unittest

from src.state.client_context import ClientDatum, PersonalInfo, TripDetails


class ClientDatumRequiredFieldsTest(unittest.TestCase):
    def test_missing_fields_ignore_interest_profile(self) -> None:
        client = ClientDatum(
            personal_info=PersonalInfo(
                name="Aisha Tan",
                email_address="aisha@example.com",
                phone_number="+6598765432",
                date_of_birth=date(1991, 6, 15),
                place_of_residence="Singapore",
                passport_number="E1234567",
            ),
            trips=[
                TripDetails(
                    destination="Bali",
                    start_date=date(2025, 12, 1),
                    end_date=date(2025, 12, 10),
                    trip_type="single",
                    trip_cost=1800.0,
                )
            ],
        )

        self.assertEqual(client.required_missing_fields(), [])


if __name__ == "__main__":
    unittest.main()
