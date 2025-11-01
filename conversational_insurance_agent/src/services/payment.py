from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import httpx
import stripe

from ..config import get_settings
from ..utils.logging import logger


class PaymentGateway:
    def __init__(self) -> None:
        self._settings = get_settings()
        if self._settings.stripe_api_key:
            stripe.api_key = self._settings.stripe_api_key
            self._configure_stripe_http_client()

    async def create_checkout_session(
        self,
        *,
        plan_code: str,
        amount: int,
        currency: str,
        success_url: str,
        cancel_url: str,
        customer_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "plan_code": plan_code,
            "amount": amount,
            "currency": currency,
            "success_url": success_url,
            "cancel_url": cancel_url,
            "customer_email": customer_email,
            "metadata": metadata or {},
        }

        logger.info("payments.create_session", plan_code=plan_code, amount=amount)

        # Attempt via auxiliary payments service first
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self._settings.payments_base_url}/payments/session",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "provider": data.get("provider", "stripe"),
                    "session_id": data["session_id"],
                    "checkout_url": data["checkout_url"],
                }
        except Exception as exc:  # noqa: BLE001 - bubble up after fallback
            logger.warning("payments.session_service_failed", error=str(exc))

        if not self._settings.stripe_api_key:
            raise RuntimeError("Unable to create payment session without Stripe credentials")

        line_items = [
            {
                "price_data": {
                    "currency": currency,
                    "product_data": {"name": plan_code},
                    "unit_amount": amount,
                },
                "quantity": 1,
            }
        ]

        session = await self._checkout_session_call(
            "create",
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            customer_email=customer_email,
            metadata=metadata or {},
        )

        return {
            "provider": "stripe",
            "session_id": session.id,
            "checkout_url": session.url,
        }

    async def fetch_status(self, session_id: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._settings.payment_status_url}/{session_id}",
                )
                if response.status_code == 404:
                    raise LookupError("Payment session not found")
                response.raise_for_status()
                return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("payments.fetch_status_fallback", error=str(exc))

        if not self._settings.stripe_api_key:
            raise RuntimeError("Cannot fetch status without service endpoint or Stripe API key")

        session = await self._checkout_session_call("retrieve", id=session_id)
        return {"session_id": session.id, "status": session.status, "payment_status": session.payment_status}

    def _configure_stripe_http_client(self) -> None:
        client = None

        try:
            from stripe.http_client import AsyncioClient  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - compatibility shim
            pass
        else:
            try:
                client = AsyncioClient()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("payments.configure_async_client_failed", error=str(exc))
                client = None

        if client is None:
            new_default_http_client = getattr(stripe, "new_default_http_client", None)
            try:
                from stripe import _http_client as stripe_http_client  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - module absent in older sdks
                stripe_http_client = None

            fallback_factory = (
                getattr(stripe_http_client, "new_http_client_async_fallback", None)
                if stripe_http_client is not None
                else None
            )

            if callable(new_default_http_client) and callable(fallback_factory):
                try:
                    signature = inspect.signature(new_default_http_client)
                except (TypeError, ValueError):
                    supports_async_kw = False
                else:
                    supports_async_kw = "async_fallback_client" in signature.parameters
                if supports_async_kw:
                    try:
                        client = new_default_http_client(
                            async_fallback_client=fallback_factory()
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("payments.configure_httpx_client_failed", error=str(exc))
                        client = None

        if client is not None:
            stripe.default_http_client = client
        else:
            logger.warning("payments.async_http_client_unconfigured")

    async def _checkout_session_call(self, method_name: str, **kwargs: Any) -> Any:
        session_cls = stripe.checkout.Session
        async_method_name = f"{method_name}_async"

        if hasattr(session_cls, async_method_name):
            method = getattr(session_cls, async_method_name)
            result = method(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        method = getattr(session_cls, method_name)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
