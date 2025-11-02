from src.web.mock_db import authenticate_user, get_user


def test_authenticate_user_success() -> None:
    record = authenticate_user("alice@example.com", "travel123")
    assert record is not None
    assert record.display_name == "Alice Tan"
    assert record.build_client().personal_info.email_address == "alice@example.com"


def test_authenticate_user_failure() -> None:
    assert authenticate_user("alice@example.com", "wrong") is None
    assert authenticate_user("unknown@example.com", "travel123") is None


def test_get_user_case_insensitive() -> None:
    assert get_user("ALICE@EXAMPLE.COM") is not None
