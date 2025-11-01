from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents import PolicyResearchAgent
from ..services.payment import PaymentGateway
from ..tools.claims_insights import ClaimsInsightTool
from ..tools.document_intelligence import DocumentIntelligenceTool
from ..utils.logging import logger
from .orchestrator import ConversationalOrchestrator
from .tooling import ToolSpec


def build_orchestrator() -> ConversationalOrchestrator:
    policy_agent = PolicyResearchAgent()
    claims_tool = ClaimsInsightTool()
    doc_tool = DocumentIntelligenceTool()
    payment_gateway = PaymentGateway()

    tools: List[ToolSpec] = [
        ToolSpec(
            name="policy_research",
            description="Agentic policy researcher that maps recommended products to eligible benefits.",
            schema={
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "Latest user request the agent should address",
                    },
                    "recommended_products": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of products the user is eligible for",
                    },
                    "tiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Corresponding tier labels for each product",
                    },
                    "chat_history": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string"},
                                "message": {"type": "string"},
                            },
                            "required": ["speaker", "message"],
                        },
                        "description": "Optional recent conversation snippets to provide context",
                    },
                },
                "required": ["user_query", "recommended_products", "tiers"],
            },
            handler=lambda user_query, recommended_products, tiers, chat_history=None: _run_policy_agent(
                policy_agent,
                user_query=user_query,
                recommended_products=recommended_products,
                tiers=tiers,
                chat_history=chat_history,
            ),
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


def _run_policy_agent(
    agent: PolicyResearchAgent,
    *,
    user_query: str,
    recommended_products: List[str] | Sequence[str],
    tiers: List[str] | Sequence[str],
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    if isinstance(recommended_products, (str, bytes)):
        recommended_products = [recommended_products]
    if isinstance(tiers, (str, bytes)):
        tiers = [tiers]

    history_tuples: List[Tuple[str, str]] = []
    if chat_history:
        for entry in chat_history:
            if isinstance(entry, dict):
                speaker = str(entry.get("speaker") or entry.get("role") or "unknown")
                message = str(entry.get("message") or entry.get("content") or "")
                history_tuples.append((speaker, message))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                speaker = str(entry[0])
                message = str(entry[1])
                history_tuples.append((speaker, message))

    result = agent.run(
        user_query=user_query,
        recommended_products=list(recommended_products),
        tiers=list(tiers),
        chat_history=history_tuples,
    )

    payload = {
        "products": result.products,
        "reasoning": result.reasoning,
        "raw": result.raw,
    }

    logger.info(
        "policy_research_agent.completed",
        products=len(result.products),
        has_reasoning=bool(result.reasoning),
    )

    return payload
