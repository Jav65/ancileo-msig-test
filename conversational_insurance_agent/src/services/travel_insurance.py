"""Ancileo travel insurance quotation and purchase integration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import re
from datetime import date, datetime

import httpx

from ..config import Settings, get_settings
from ..utils.logging import logger


class AncileoAPIError(RuntimeError):
    """Raised when the Ancileo platform returns an error response."""


class AncileoTravelAPI:
    """Wrapper around Ancileo's travel insurance APIs used in the hackathon."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = (self._settings.ancileo_base_url or "").rstrip("/")
        if not self._base_url:
            raise ValueError("ANCILEO_BASE_URL is not configured")
        self._timeout = 15.0

    async def quote(self, **payload: Any) -> Dict[str, Any]:
        """Call the Ancileo quotation endpoint and return the JSON payload.

        The integration currently requires a fixed payload provided by the
        stakeholder, so any payload supplied by the caller is ignored in favour
        of that hard-coded request.
        """

        request = self._hardcoded_quote_payload()
        data = await self._post("/pricing", request)

        quote_id = data.get("quoteId") if isinstance(data, dict) else None
        offers = data.get("offers") if isinstance(data, dict) else None
        offers_count = len(offers) if isinstance(offers, list) else 0
        logger.info("ancileo.quote.success", quote_id=quote_id, offers=offers_count)

        return data

    @staticmethod
    def _hardcoded_quote_payload() -> Dict[str, Any]:
        """Return the fixed quotation payload expected by the Ancileo sandbox."""

        return {
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

    async def purchase(self, **payload: Any) -> Dict[str, Any]:
        """Call the Ancileo purchase endpoint after successful payment."""

        request = self._prepare_purchase_payload(payload)
        data = await self._post("/purchase", request)

        logger.info(
            "ancileo.purchase.success",
            quote_id=request.get("quoteId"),
            offers=len(request.get("purchaseOffers", [])),
        )

        return data

    # ------------------------------------------------------------------
    # Payload preparation helpers
    # ------------------------------------------------------------------
    def _prepare_quote_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Quote payload must be a JSON object")

        request: Dict[str, Any] = {
            "market": self._coerce_str(payload.get("market"))
            or self._settings.ancileo_default_market,
            "languageCode": self._coerce_str(payload.get("languageCode"))
            or self._settings.ancileo_default_language,
            "channel": self._coerce_str(payload.get("channel"))
            or self._settings.ancileo_default_channel,
        }

        request["deviceType"] = self._normalize_device_type(payload.get("deviceType"))

        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError("Quote payload must include a 'context' object")

        normalized_context = self._normalize_quote_context(context)
        request["context"] = normalized_context

        return request

    def _normalize_quote_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trip_type = self._normalize_trip_type(context.get("tripType"))
        if trip_type is None:
            raise ValueError("context.tripType must be 'ST'/'RT' or a recognisable trip type")

        departure_date = self._normalize_date_value(context.get("departureDate"), field="departureDate")
        arrival_country = self._normalize_country_code(context.get("arrivalCountry"), field="arrivalCountry")
        departure_country_raw = context.get("departureCountry") or self._settings.ancileo_default_market
        departure_country = self._normalize_country_code(departure_country_raw, field="departureCountry")

        adults_count = self._coerce_int(context.get("adultsCount"), minimum=1, field="adultsCount")
        children_count = self._coerce_int(context.get("childrenCount"), minimum=0, field="childrenCount", default=0)

        normalized: Dict[str, Any] = {
            "tripType": trip_type,
            "departureDate": departure_date,
            "departureCountry": departure_country,
            "arrivalCountry": arrival_country,
            "adultsCount": adults_count,
            "childrenCount": children_count,
        }

        if trip_type == "RT":
            return_date = self._normalize_date_value(context.get("returnDate"), field="returnDate")
            normalized["returnDate"] = return_date
        elif context.get("returnDate"):
            # Allow callers to submit returnDate for ST trips; API tolerates it but we normalise casing
            normalized["returnDate"] = self._normalize_date_value(context.get("returnDate"), field="returnDate")

        return normalized

    def _prepare_purchase_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Purchase payload must be a JSON object")

        request: Dict[str, Any] = {
            "market": self._coerce_str(payload.get("market"))
            or self._settings.ancileo_default_market,
            "languageCode": self._coerce_str(payload.get("languageCode"))
            or self._settings.ancileo_default_language,
            "channel": self._coerce_str(payload.get("channel"))
            or self._settings.ancileo_default_channel,
            "quoteId": self._require_str(payload, "quoteId"),
        }

        purchase_offers = payload.get("purchaseOffers")
        if not isinstance(purchase_offers, list) or not purchase_offers:
            raise ValueError("purchaseOffers must be a non-empty array")
        request["purchaseOffers"] = [self._normalize_purchase_offer(item) for item in purchase_offers]

        insureds = payload.get("insureds")
        if not isinstance(insureds, list) or not insureds:
            raise ValueError("insureds must be a non-empty array")
        request["insureds"] = [self._normalize_insured(item) for item in insureds]

        main_contact = payload.get("mainContact")
        if not isinstance(main_contact, dict):
            raise ValueError("mainContact must be an object containing traveller contact details")
        request["mainContact"] = self._normalize_main_contact(main_contact)

        return request

    def _normalize_purchase_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(offer, dict):
            raise ValueError("Each purchase offer must be an object")

        normalized = {
            "productType": self._require_str(offer, "productType"),
            "offerId": self._require_str(offer, "offerId"),
            "productCode": self._require_str(offer, "productCode"),
            "unitPrice": self._coerce_float(offer.get("unitPrice"), field="unitPrice"),
            "currency": self._require_str(offer, "currency"),
            "quantity": self._coerce_int(offer.get("quantity"), minimum=1, field="quantity"),
            "totalPrice": self._coerce_float(offer.get("totalPrice"), field="totalPrice"),
            "isSendEmail": self._coerce_bool(offer.get("isSendEmail"), default=True),
        }

        return normalized

    def _normalize_insured(self, insured: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(insured, dict):
            raise ValueError("Each insured entry must be an object")

        required_fields = [
            "id",
            "title",
            "firstName",
            "lastName",
            "nationality",
            "dateOfBirth",
            "passport",
            "email",
            "phoneType",
            "phoneNumber",
            "relationship",
        ]

        normalized = {field: self._require_str(insured, field) for field in required_fields}

        return normalized

    def _normalize_main_contact(self, contact: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_insured(contact)

        extra_fields = {
            "address": self._require_str(contact, "address"),
            "city": self._require_str(contact, "city"),
            "zipCode": self._require_str(contact, "zipCode"),
            "countryCode": self._require_str(contact, "countryCode"),
        }

        normalized.update(extra_fields)
        return normalized

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_api_key()

        url = f"{self._base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._settings.ancileo_api_key,
        }

        logger.info(
            "ancileo.request",
            endpoint=endpoint,
            summary=self._summarize_payload(payload),
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - exercised via unit tests
            body = self._safe_text(exc.response)
            logger.error(
                "ancileo.http_error",
                endpoint=endpoint,
                status=exc.response.status_code,
                body=body,
            )
            raise AncileoAPIError(
                f"Ancileo API returned {exc.response.status_code}: {body or 'unknown error'}"
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network failures
            logger.error("ancileo.transport_error", endpoint=endpoint, error=str(exc))
            raise AncileoAPIError("Unable to reach Ancileo API") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            logger.error("ancileo.invalid_json", endpoint=endpoint)
            raise AncileoAPIError("Ancileo API returned invalid JSON") from exc

        return data

    def _ensure_api_key(self) -> None:
        if not self._settings.ancileo_api_key:
            raise AncileoAPIError("ANCILEO_API_KEY is not configured")

    # ------------------------------------------------------------------
    # Type coercion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _normalize_device_type(self, value: Any) -> str:
        raw = self._coerce_str(value)
        if not raw:
            raw = self._settings.ancileo_default_device

        cleaned = re.sub(r"[\s_-]+", " ", raw).strip().upper()
        cleaned = cleaned.replace(" ", "")

        aliases = {
            "SMARTPHONE": "MOBILE",
            "PHONE": "MOBILE",
            "CELL": "MOBILE",
            "LAPTOP": "DESKTOP",
            "PC": "DESKTOP",
            "TABLETPC": "TABLET",
        }

        normalized = aliases.get(cleaned, cleaned)

        allowed = {"DESKTOP", "MOBILE", "TABLET", "OTHER"}
        if normalized not in allowed:
            raise ValueError("deviceType must be one of DESKTOP/MOBILE/TABLET/OTHER")

        return normalized

    def _normalize_date_value(self, value: Any, *, field: str) -> str:
        text = self._coerce_str(value)
        if not text:
            raise ValueError(f"Field '{field}' is required and cannot be empty")

        normalized = self._coerce_date_string(text)
        if normalized is None:
            raise ValueError(f"Field '{field}' must be a valid date in YYYY-MM-DD format")

        return normalized

    def _normalize_country_code(self, value: Any, *, field: str) -> str:
        text = self._coerce_str(value)
        if not text:
            raise ValueError(f"Field '{field}' is required and cannot be empty")

        normalized = text.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2}", normalized):
            raise ValueError(f"Field '{field}' must be a valid ISO country code")

        return normalized

    @staticmethod
    def _require_str(payload: Dict[str, Any], field: str) -> str:
        value = payload.get(field)
        text = AncileoTravelAPI._coerce_str(value)
        if not text:
            raise ValueError(f"Field '{field}' is required and cannot be empty")
        return text

    @staticmethod
    def _coerce_int(value: Any, *, minimum: int, field: str, default: Optional[int] = None) -> int:
        if value is None:
            if default is not None:
                value = default
            else:
                raise ValueError(f"Field '{field}' is required")

        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{field}' must be an integer") from exc

        if numeric < minimum:
            raise ValueError(f"Field '{field}' must be at least {minimum}")

        return numeric

    @staticmethod
    def _coerce_float(value: Any, *, field: str) -> float:
        if value is None:
            raise ValueError(f"Field '{field}' is required")

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{field}' must be numeric") from exc

        return numeric

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _normalize_trip_type(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if text in {"st", "single", "single_trip", "one_way", "one-way"}:
            return "ST"
        if text in {"rt", "round", "round_trip", "roundtrip", "return"}:
            return "RT"
        return None

    @staticmethod
    def _coerce_date_string(value: str) -> Optional[str]:
        candidate = value.strip()
        if not candidate:
            return None

        # Normalise common separators
        candidate = candidate.replace("/", "-").replace(".", "-")
        candidate = re.sub(r"\s+", " ", candidate)

        # Direct YYYY-MM-DD or YYYY-M-D
        match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", candidate)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

        iso_candidate = candidate
        if iso_candidate.endswith("Z"):
            iso_candidate = iso_candidate[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(iso_candidate)
            return parsed.date().isoformat()
        except ValueError:
            pass

        # Additional relaxed formats
        patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d %b %Y",
            "%d %B %Y",
            "%b %d %Y",
            "%B %d %Y",
        ]

        for pattern in patterns:
            try:
                parsed = datetime.strptime(candidate, pattern)
                return parsed.date().isoformat()
            except ValueError:
                continue

        return None

    @staticmethod
    def _safe_text(response: httpx.Response) -> str:
        try:
            return response.text
        except Exception:  # pragma: no cover - defensive
            return ""

    @staticmethod
    def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key in ("market", "languageCode", "channel", "deviceType", "quoteId"):
            if key in payload:
                summary[key] = payload[key]

        context = payload.get("context")
        if isinstance(context, dict):
            summary["context"] = {
                key: context.get(key)
                for key in (
                    "tripType",
                    "departureDate",
                    "returnDate",
                    "departureCountry",
                    "arrivalCountry",
                    "adultsCount",
                    "childrenCount",
                )
            }

        purchase_offers = payload.get("purchaseOffers")
        if isinstance(purchase_offers, Iterable):
            summary["purchaseOffers"] = len(list(purchase_offers))

        insureds = payload.get("insureds")
        if isinstance(insureds, Iterable):
            summary["insureds"] = len(list(insureds))

        return summary

