from __future__ import annotations

import json
import os
from datetime import date

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Ensure required configuration is present before importing the app
os.environ.setdefault("GROQ_API_KEY", "test-key")

from src.main import app
from src.gmail_portal import router
from src.config import Settings
from src.state.client_context import ClientDatum, TripDetails


@pytest.fixture(autouse=True)
def reset_settings_cache():
    from src import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_chat_page_handles_oauth_callback_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_complete(request, settings):  # type: ignore[no-untyped-def]
        calls.append((request.query_params.get("code"), request.query_params.get("state")))

    monkeypatch.setattr(router, "_complete_oauth_login", fake_complete)

    with TestClient(app) as client:
        response = client.get("/gmail/chat?code=dummy-code&state=dummy-state", allow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/gmail/chat"
    assert calls == [("dummy-code", "dummy-state")]


def test_build_flow_enables_insecure_transport_for_local_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OAUTHLIB_INSECURE_TRANSPORT", raising=False)
    settings = Settings(
        groq_api_key="test-key",
        google_client_id="client-id",
        google_client_secret="client-secret",
        google_redirect_uri="http://localhost:8000/gmail/callback",
    )

    flow = router._build_flow(settings)

    assert flow.redirect_uri == settings.google_redirect_uri
    assert os.environ["OAUTHLIB_INSECURE_TRANSPORT"] == "1"


def test_build_flow_rejects_non_local_insecure_redirect() -> None:
    settings = Settings(
        groq_api_key="test-key",
        google_client_id="client-id",
        google_client_secret="client-secret",
        google_redirect_uri="http://example.com/callback",
    )

    with pytest.raises(HTTPException) as exc_info:
        router._build_flow(settings)

    exc = exc_info.value
    assert exc.status_code == 500
    assert "plain HTTP" in exc.detail


def test_gmail_client_serialization_is_json_safe() -> None:
    client = ClientDatum(
        trips=[
            TripDetails(
                destination="Tokyo",
                start_date=date(2024, 5, 1),
                end_date=date(2024, 5, 10),
            )
        ]
    )

    payload = router._serialize_client(client)

    assert isinstance(payload["trips"], list)
    assert payload["trips"][0]["startDate"] == "2024-05-01"
    assert payload["trips"][0]["endDate"] == "2024-05-10"

    json.dumps(payload)  # Should not raise TypeError
