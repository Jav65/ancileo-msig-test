from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..agents import PolicyResearchAgent
from ..services.payment import PaymentGateway
from ..services.travel_insurance import AncileoTravelAPI
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
    ancileo_api = AncileoTravelAPI()

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
            handler=partial(_policy_research_tool_handler, policy_agent),
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
        # Ancileo real-time pricing is disabled; LLM now drives quotes directly from conversation context.
        # ToolSpec(
        #     name="travel_insurance_quote",
        #     description=(
        #         "Fetch live travel insurance quotation from the Ancileo API. "
        #         "Always call this tool to obtain premiums before sharing price information."
        #     ),
        #     schema={
        #         "type": "object",
        #         "properties": {
        #             "market": {
        #                 "type": "string",
        #                 "description": "Market code such as 'SG'. Defaults to ANCILEO_DEFAULT_MARKET.",
        #             },
        #             "languageCode": {
        #                 "type": "string",
        #                 "description": "Language preference, e.g. 'en'. Defaults to ANCILEO_DEFAULT_LANGUAGE.",
        #             },
        #             "channel": {
        #                 "type": "string",
        #                 "description": "Distribution channel identifier, defaults to ANCILEO_DEFAULT_CHANNEL.",
        #             },
        #             "deviceType": {
        #                 "type": "string",
        #                 "description": "Device context such as 'DESKTOP'. Defaults to ANCILEO_DEFAULT_DEVICE.",
        #             },
        #             "context": {
        #                 "type": "object",
        #                 "description": "Trip context required by the pricing endpoint.",
        #                 "properties": {
        #                     "tripType": {
        #                         "type": "string",
        #                         "description": "'ST' for single trip or 'RT' for round trip (case insensitive variants accepted).",
        #                     },
        #                     "departureDate": {
        #                         "type": "string",
        #                         "description": "Departure date in YYYY-MM-DD format.",
        #                     },
        #                     "returnDate": {
        #                         "type": "string",
        #                         "description": "Return date in YYYY-MM-DD format (required for round trips).",
        #                     },
        #                     "departureCountry": {
        #                         "type": "string",
        #                         "description": "ISO country code where the trip starts.",
        #                     },
        #                     "arrivalCountry": {
        #                         "type": "string",
        #                         "description": "ISO country code of the destination.",
        #                     },
        #                     "adultsCount": {
        #                         "type": "integer",
        #                         "description": "Number of adults travelling (must be >= 1).",
        #                         "minimum": 1,
        #                     },
        #                     "childrenCount": {
        #                         "type": "integer",
        #                         "description": "Number of children travelling (defaults to 0).",
        #                         "minimum": 0,
        #                     },
        #                 },
        #                 "required": [
        #                     "tripType",
        #                     "departureDate",
        #                     "departureCountry",
        #                     "arrivalCountry",
        #                     "adultsCount",
        #                 ],
        #             },
        #         },
        #         "required": ["context"],
        #     },
        #     handler=ancileo_api.quote,
        #     is_async=True,
        # ),
        ToolSpec(
            name="travel_insurance_purchase",
            description=(
                "Complete the policy issuance with the Ancileo purchase API after confirming payment. "
                "Use the quoteId/offerId gathered during the conversation together with traveller identity data."
            ),
            schema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Market code (defaults to ANCILEO_DEFAULT_MARKET).",
                    },
                    "languageCode": {
                        "type": "string",
                        "description": "Language preference (defaults to ANCILEO_DEFAULT_LANGUAGE).",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Distribution channel (defaults to ANCILEO_DEFAULT_CHANNEL).",
                    },
                    "quoteId": {
                        "type": "string",
                        "description": "Quote UUID returned from the pricing step.",
                    },
                    "purchaseOffers": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "productType": {"type": "string"},
                                "offerId": {"type": "string"},
                                "productCode": {"type": "string"},
                                "unitPrice": {"type": "number"},
                                "currency": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                                "totalPrice": {"type": "number"},
                                "isSendEmail": {"type": "boolean"},
                            },
                            "required": [
                                "productType",
                                "offerId",
                                "productCode",
                                "unitPrice",
                                "currency",
                                "quantity",
                                "totalPrice",
                            ],
                        },
                    },
                    "insureds": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "firstName": {"type": "string"},
                                "lastName": {"type": "string"},
                                "nationality": {"type": "string"},
                                "dateOfBirth": {"type": "string"},
                                "passport": {"type": "string"},
                                "email": {"type": "string"},
                                "phoneType": {"type": "string"},
                                "phoneNumber": {"type": "string"},
                                "relationship": {"type": "string"},
                            },
                            "required": [
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
                            ],
                        },
                    },
                    "mainContact": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "firstName": {"type": "string"},
                            "lastName": {"type": "string"},
                            "nationality": {"type": "string"},
                            "dateOfBirth": {"type": "string"},
                            "passport": {"type": "string"},
                            "email": {"type": "string"},
                            "phoneType": {"type": "string"},
                            "phoneNumber": {"type": "string"},
                            "address": {"type": "string"},
                            "city": {"type": "string"},
                            "zipCode": {"type": "string"},
                            "countryCode": {"type": "string"},
                        },
                        "required": [
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
                            "address",
                            "city",
                            "zipCode",
                            "countryCode",
                        ],
                    },
                },
                "required": ["quoteId", "purchaseOffers", "insureds", "mainContact"],
            },
            handler=ancileo_api.purchase,
            is_async=True,
        ),
        ToolSpec(
            name="payment_checkout",
            description=(
                "Create and monitor a payment checkout session for purchasing a travel insurance plan. "
                "Provide the plan_code, price, and metadata as determined by the LLM-guided consultation."
            ),
            schema={
                "type": "object",
                "properties": {
                    "plan_code": {
                        "type": "string",
                        "description": "Use the pricing offer's productCode as the plan identifier.",
                    },
                    "amount": {"type": "integer", "description": "Amount in minor currency units"},
                    "currency": {"type": "string", "default": "sgd"},
                    "success_url": {"type": "string"},
                    "cancel_url": {"type": "string"},
                    "customer_email": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "description": "Additional context such as quoteId, offerId, productCode, traveller info.",
                    },
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


def _policy_research_tool_handler(
    agent: PolicyResearchAgent,
    *,
    user_query: str,
    recommended_products: Sequence[str] | str | None = None,
    tiers: Sequence[str] | str | None = None,
    chat_history: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    missing_fields: List[str] = []
    if recommended_products is None:
        missing_fields.append("recommended_products")
        recommended_products = []
    if tiers is None:
        missing_fields.append("tiers")
        tiers = []

    if missing_fields:
        logger.warning(
            "policy_research_agent.missing_tool_arguments",
            missing=missing_fields,
            user_query_preview=(user_query or "")[:80],
        )

    return _run_policy_agent(
        agent,
        user_query=user_query,
        recommended_products=recommended_products,
        tiers=tiers,
        chat_history=chat_history,
    )


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
