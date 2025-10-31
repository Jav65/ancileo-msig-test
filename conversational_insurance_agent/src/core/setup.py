from __future__ import annotations

from typing import List

from ..services.payment import PaymentGateway
from ..tools.claims_insights import ClaimsInsightTool
from ..tools.document_intelligence import DocumentIntelligenceTool
from ..tools.policy_rag import PolicyRAGTool
from ..utils.logging import logger
from .orchestrator import ConversationalOrchestrator
from .tooling import ToolSpec


def build_orchestrator() -> ConversationalOrchestrator:
    policy_tool = PolicyRAGTool()
    claims_tool = ClaimsInsightTool()
    doc_tool = DocumentIntelligenceTool()
    payment_gateway = PaymentGateway()

    try:
        policy_tool.ensure_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("policy_rag.ensure_index_failed", error=str(exc))

    tools: List[ToolSpec] = [
        ToolSpec(
            name="policy_lookup",
            description="Retrieve relevant policy passages with citations for a user question.",
            schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User query about coverage"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
                },
                "required": ["question"],
            },
            handler=policy_tool.query,
        ),
        ToolSpec(
            name="claims_recommendation",
            description="Generate plan recommendations and risk insights using historical claims data.",
            schema={
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Trip destination"},
                    "activity": {"type": "string", "description": "Primary trip activity"},
                    "trip_cost": {"type": "number", "description": "Estimated total trip cost in currency units"},
                },
            },
            handler=lambda destination=None, activity=None, trip_cost=None: claims_tool.recommend_plan(
                destination=destination,
                activity=activity,
                trip_cost=trip_cost,
            ),
        ),
        ToolSpec(
            name="document_ingest",
            description="Extract traveler, destination, and date data from an uploaded itinerary or booking.",
            schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the document staged by the channel adapter",
                    }
                },
                "required": ["file_path"],
            },
            handler=lambda file_path: doc_tool.parse_trip_document(file_path=file_path),
        ),
        ToolSpec(
            name="payment_checkout",
            description="Create and monitor a payment checkout session for purchasing a travel insurance plan.",
            schema={
                "type": "object",
                "properties": {
                    "plan_code": {"type": "string"},
                    "amount": {"type": "integer", "description": "Amount in minor currency units"},
                    "currency": {"type": "string", "default": "sgd"},
                    "success_url": {"type": "string"},
                    "cancel_url": {"type": "string"},
                    "customer_email": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["plan_code", "amount", "currency", "success_url", "cancel_url"],
            },
            handler=payment_gateway.create_checkout_session,
            is_async=True,
        ),
        ToolSpec(
            name="payment_status",
            description="Retrieve the latest status of a previously created payment session.",
            schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Stripe or platform session ID"}
                },
                "required": ["session_id"],
            },
            handler=payment_gateway.fetch_status,
            is_async=True,
        ),
    ]

    return ConversationalOrchestrator(tools)
