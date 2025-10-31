from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
import stripe
from stripe.http_client import AsyncioClient

from ..config import get_settings
from ..utils.logging import logger


class PaymentGateway:
    def __init__(self) -> None:
        self._settings = get_settings()
        if self._settings.stripe_api_key:
            stripe.api_key = self._settings.stripe_api_key
            stripe.default_http_client = AsyncioClient()

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

        session = await stripe.checkout.Session.create(
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

        session = await stripe.checkout.Session.retrieve(session_id)
        return {"session_id": session.id, "status": session.status, "payment_status": session.payment_status}
