from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure required configuration is present before importing the app
os.environ.setdefault("GROQ_API_KEY", "test-key")

from src.main import app
from src.gmail_portal import router


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
