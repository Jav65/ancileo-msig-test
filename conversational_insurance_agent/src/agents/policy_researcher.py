"""Agent responsible for researching policy benefits using taxonomy data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..utils.logging import logger


class AgentState(TypedDict, total=False):
    """Internal state tracked by the policy research agent."""

    user_query: str
    chat_history: Sequence[Tuple[str, str]]
    recommended_products: Sequence[str]
    tiers: Sequence[str]
    taxonomy_context: str
    eligible_benefits_of_products: Any
    llm_raw_output: Optional[str]


@dataclass(frozen=True)
class PolicyResearchResult:
    """Outcome returned by the policy research agent."""

    products: List[Dict[str, Any]]
    reasoning: Optional[str]
    raw: Optional[str]


class PolicyResearchAgent:
    """LangGraph-powered agent that surfaces policy benefits for recommended products."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatGroq] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._taxonomy_path = self._settings.taxonomy_path
        self._taxonomy_payload: Dict[str, Any] = {}
        self._taxonomy_mtime: float | None = None
        self._load_taxonomy_payload()
        self._llm = llm or ChatGroq(
            model=self._settings.groq_model,
            temperature=0.2,
            groq_api_key=self._settings.groq_api_key,
        )
        self._graph = self._build_graph()

    def run(
        self,
        *,
        user_query: str,
        recommended_products: Sequence[str],
        tiers: Sequence[str],
        chat_history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> PolicyResearchResult:
        """Execute the agent and return the researched benefits."""

        self._ensure_taxonomy_fresh()

        initial_state: AgentState = {
            "user_query": user_query,
            "chat_history": chat_history or [],
            "recommended_products": recommended_products,
            "tiers": tiers,
            "eligible_benefits_of_products": None,
        }

        final_state = self._graph.invoke(initial_state)

        benefits = cast(
            List[Dict[str, Any]],
            final_state.get("eligible_benefits_of_products") or [],
        )

        reasoning: Optional[str] = None
        raw_output = final_state.get("llm_raw_output")
        if isinstance(raw_output, str):
            try:
                payload = json.loads(raw_output)
                reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                reasoning = None

        return PolicyResearchResult(products=benefits, reasoning=reasoning, raw=raw_output)

    # ---------------------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        graph: StateGraph = StateGraph(AgentState)
        graph.add_node("prepare_context", self._prepare_context)
        graph.add_node("llm_reasoning", self._llm_reasoning)

        graph.set_entry_point("prepare_context")
        graph.add_edge("prepare_context", "llm_reasoning")
        graph.add_edge("llm_reasoning", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # State nodes
    # ------------------------------------------------------------------
    def _prepare_context(self, state: AgentState) -> AgentState:
        products = list(state.get("recommended_products") or [])
        tiers = list(state.get("tiers") or [])

        if not products:
            logger.info("policy_research_agent.no_products", user_query=state.get("user_query"))
            state["taxonomy_context"] = ""
            return state

        taxonomy_text = self._render_taxonomy_context(products, tiers)
        state["taxonomy_context"] = taxonomy_text
        return state

    def _llm_reasoning(self, state: AgentState) -> AgentState:
        context = state.get("taxonomy_context", "")
        products = list(state.get("recommended_products") or [])

        if not products or not context.strip():
            logger.warning(
                "policy_research_agent.skipping_llm",
                reason="missing_products_or_context",
            )
            state["eligible_benefits_of_products"] = []
            state["llm_raw_output"] = None
            return state

        tiers = list(state.get("tiers") or [])
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history") or []

        prompt = self._build_prompt(
            user_query=user_query,
            chat_history=chat_history,
            context=context,
            products=products,
            tiers=tiers,
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a travel insurance policy researcher. "
                    "Review the supplied taxonomy carefully. "
                    "Only return benefits when the taxonomy indicates eligibility. "
                    "Always respond with valid JSON matching the requested schema."
                )
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages, response_format={"type": "json_object"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("policy_research_agent.llm_failure", error=str(exc))
            state["eligible_benefits_of_products"] = []
            state["llm_raw_output"] = json.dumps(
                {
                    "error": "llm_failure",
                    "message": str(exc),
                }
            )
            return state

        output_text = ""
        if hasattr(response, "content"):
            output_text = cast(str, response.content)
        elif isinstance(response, str):
            output_text = response
        else:
            output_text = str(response)

        parsed = self._safe_parse_json(output_text)
        products_payload = []
        if isinstance(parsed, dict):
            raw_products = parsed.get("products")
            if isinstance(raw_products, list):
                products_payload = [item for item in raw_products if isinstance(item, dict)]

        state["eligible_benefits_of_products"] = products_payload
        state["llm_raw_output"] = output_text
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_taxonomy_context(
        self,
        products: Sequence[str],
        tiers: Sequence[str],
    ) -> str:
        layers = self._taxonomy_payload.get("layers", {}) if isinstance(self._taxonomy_payload, dict) else {}
        general_conditions = layers.get("layer_1_general_conditions", []) if isinstance(layers, dict) else []
        benefits = layers.get("layer_2_benefits", []) if isinstance(layers, dict) else []
        benefit_conditions: Any = []
        if isinstance(layers, dict):
            benefit_conditions = layers.get("layer_3_benefit_specific_conditions")
            if not benefit_conditions:
                benefit_conditions = layers.get("layer_3_benefit_conditions", [])

        product_sections: List[str] = []
        for index, product in enumerate(products):
            tier = tiers[index] if index < len(tiers) else ""
            section: Dict[str, Any] = {
                "product": product,
                "tier": tier,
                "general_conditions": self._filter_product_entries(general_conditions, product, key="condition"),
                "benefits": self._filter_product_entries(benefits, product, key="benefit_name"),
                "benefit_conditions": self._filter_product_entries(
                    benefit_conditions,
                    product,
                    key="condition",
                ),
            }
            product_sections.append(yaml.dump(section, sort_keys=False, default_flow_style=False))

        return "\n---\n".join(product_sections)

    @staticmethod
    def _filter_product_entries(entries: Any, product: str, key: str) -> List[Dict[str, Any]]:
        if not isinstance(entries, list):
            return []

        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            products_data = entry.get("products")
            if not isinstance(products_data, dict):
                continue
            product_payload = products_data.get(product)
            if not product_payload:
                continue
            filtered.append(
                {
                    key: entry.get(key),
                    "details": product_payload,
                    "parameters": entry.get("parameters"),
                    "condition_type": entry.get("condition_type"),
                }
            )

        return filtered

    @staticmethod
    def _build_prompt(
        *,
        user_query: str,
        chat_history: Sequence[Tuple[str, str]],
        context: str,
        products: Sequence[str],
        tiers: Sequence[str],
    ) -> str:
        history_lines: List[str] = []
        for speaker, message in chat_history:
            history_lines.append(f"{speaker}: {message}")
        history_block = "\n".join(history_lines)

        product_descriptions = []
        for index, product in enumerate(products):
            tier = tiers[index] if index < len(tiers) else ""
            product_descriptions.append(f"- {product} (tier: {tier or 'unspecified'})")
        products_block = "\n".join(product_descriptions)

        return (
            "The user has asked: "
            f"{user_query}\n\n"
            "Conversation history (most recent last):\n"
            f"{history_block or 'No previous context.'}\n\n"
            "Recommended products and tiers:\n"
            f"{products_block}\n\n"
            "Taxonomy excerpts relevant to these products:\n"
            f"{context}\n\n"
            "Please produce a JSON object with the shape:\n"
            "{\n"
            "  \"products\": [\n"
            "    {\n"
            "      \"product\": string,\n"
            "      \"tier\": string,\n"
            "      \"benefits\": [\n"
            "        {\n"
            "          \"name\": string,\n"
            "          \"why_eligible\": string,\n"
            "          \"parameters\": object | null,\n"
            "          \"conditions\": [string]\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ],\n"
            "  \"reasoning\": string\n"
            "}\n"
            "Only include benefits that the user appears eligible for."
        )

    @staticmethod
    def _safe_parse_json(payload: str) -> Any:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("policy_research_agent.parse_failure", preview=payload[:200])
            return None

    @staticmethod
    def _load_taxonomy(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Taxonomy payload should be a JSON object")
        return payload

    def _load_taxonomy_payload(self) -> None:
        payload = self._load_taxonomy(self._taxonomy_path)
        self._taxonomy_payload = payload
        try:
            self._taxonomy_mtime = self._taxonomy_path.stat().st_mtime
        except OSError:
            self._taxonomy_mtime = None

    def _ensure_taxonomy_fresh(self) -> None:
        try:
            current_mtime = self._taxonomy_path.stat().st_mtime
        except FileNotFoundError as exc:
            logger.error("policy_research_agent.taxonomy_missing", path=str(self._taxonomy_path))
            raise

        if self._taxonomy_mtime is None or current_mtime > self._taxonomy_mtime:
            logger.info("policy_research_agent.taxonomy_reload", path=str(self._taxonomy_path))
            self._load_taxonomy_payload()

