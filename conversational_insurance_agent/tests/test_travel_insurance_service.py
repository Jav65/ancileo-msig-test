from typing import Any, Dict

import pytest

from src.services.travel_insurance import AncileoAPIError, AncileoTravelAPI


class _DummySettings:
    ancileo_base_url = "https://ancileo.test/v1/travel/front"
    ancileo_api_key = "test-key"
    ancileo_default_market = "SG"
    ancileo_default_language = "en"
    ancileo_default_channel = "white-label"
    ancileo_default_device = "DESKTOP"


class _CapturingResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = "{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class _CapturingClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.captured: Dict[str, Any] = {}

    async def __aenter__(self) -> "_CapturingClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, Any]):
        self.captured = {
            "url": url,
            "json": json,
            "headers": headers,
        }
        return _CapturingResponse({"quoteId": "quote-123", "offers": []})


@pytest.mark.asyncio
async def test_quote_normalises_context_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    client_holder: Dict[str, Any] = {}

    def _client_factory(*args: Any, **kwargs: Any) -> _CapturingClient:
        client = _CapturingClient(*args, **kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr("src.services.travel_insurance.httpx.AsyncClient", _client_factory)

    api = AncileoTravelAPI(settings=_DummySettings())

    quote_payload = {
        "context": {
            "tripType": "round",
            "departureDate": "2025-09-30",
            "returnDate": "2025-10-05",
            "departureCountry": "sg",
            "arrivalCountry": "cn",
            "adultsCount": 2,
        }
    }

    data = await api.quote(**quote_payload)

    assert data["quoteId"] == "quote-123"
    captured = client_holder["client"].captured  # type: ignore[index]
    assert captured["url"] == "https://ancileo.test/v1/travel/front/pricing"
    assert captured["headers"]["x-api-key"] == "test-key"
    request_json = captured["json"]
    assert request_json["market"] == "SG"
    assert request_json["languageCode"] == "en"
    assert request_json["channel"] == "white-label"
    assert request_json["deviceType"] == "DESKTOP"
    assert request_json["context"]["tripType"] == "RT"
    assert request_json["context"]["childrenCount"] == 0
    assert request_json["context"]["arrivalCountry"] == "CN"
    assert request_json["context"]["departureCountry"] == "SG"


@pytest.mark.asyncio
async def test_quote_normalises_dates_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    client_holder: Dict[str, Any] = {}

    def _client_factory(*args: Any, **kwargs: Any) -> _CapturingClient:
        client = _CapturingClient(*args, **kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr("src.services.travel_insurance.httpx.AsyncClient", _client_factory)

    api = AncileoTravelAPI(settings=_DummySettings())

    quote_payload = {
        "deviceType": " mobile ",
        "context": {
            "tripType": "st",
            "departureDate": "1 Nov 2025",
            "returnDate": "15 Nov 2025",
            "departureCountry": "sg",
            "arrivalCountry": "cn",
            "adultsCount": "1",
            "childrenCount": "0",
        },
    }

    await api.quote(**quote_payload)

    request_json = client_holder["client"].captured["json"]  # type: ignore[index]
    assert request_json["deviceType"] == "MOBILE"
    assert request_json["context"]["departureDate"] == "2025-11-01"
    assert request_json["context"]["returnDate"] == "2025-11-15"


@pytest.mark.asyncio
async def test_quote_payload_matches_required_format(monkeypatch: pytest.MonkeyPatch) -> None:
    client_holder: Dict[str, Any] = {}

    def _client_factory(*args: Any, **kwargs: Any) -> _CapturingClient:
        client = _CapturingClient(*args, **kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr("src.services.travel_insurance.httpx.AsyncClient", _client_factory)

    api = AncileoTravelAPI(settings=_DummySettings())

    expected_payload = {
        "market": "SG",
        "languageCode": "en",
        "channel": "white-label",
        "deviceType": "DESKTOP",
        "context": {
            "tripType": "ST",
            "departureDate": "2025-11-01",
            "returnDate": "2025-11-15",
            "departureCountry": "SG",
            "arrivalCountry": "CN",
            "adultsCount": 1,
            "childrenCount": 0,
        },
    }

    await api.quote(**expected_payload)

    captured_request = client_holder["client"].captured["json"]  # type: ignore[index]
    assert captured_request == expected_payload


@pytest.mark.asyncio
async def test_purchase_requires_api_key() -> None:
    class _NoKeySettings(_DummySettings):
        ancileo_api_key = ""

    api = AncileoTravelAPI(settings=_NoKeySettings())

    purchase_payload = {
        "quoteId": "quote-1",
        "purchaseOffers": [
            {
                "productType": "travel-insurance",
                "offerId": "offer-1",
                "productCode": "AXA",
                "unitPrice": 17.6,
                "currency": "SGD",
                "quantity": 1,
                "totalPrice": 17.6,
            }
        ],
        "insureds": [
            {
                "id": "1",
                "title": "Mr",
                "firstName": "John",
                "lastName": "Doe",
                "nationality": "SG",
                "dateOfBirth": "2000-01-01",
                "passport": "123456",
                "email": "john.doe@gmail.com",
                "phoneType": "mobile",
                "phoneNumber": "081111111",
                "relationship": "main",
            }
        ],
        "mainContact": {
            "id": "1",
            "title": "Mr",
            "firstName": "John",
            "lastName": "Doe",
            "nationality": "SG",
            "dateOfBirth": "2000-01-01",
            "passport": "123456",
            "email": "john.doe@gmail.com",
            "phoneType": "mobile",
            "phoneNumber": "081111111",
            "relationship": "main",
            "address": "12 Test Street",
            "city": "SG",
            "zipCode": "12345",
            "countryCode": "SG",
        },
    }

    with pytest.raises(AncileoAPIError, match="ANCILEO_API_KEY is not configured"):
        await api.purchase(**purchase_payload)

