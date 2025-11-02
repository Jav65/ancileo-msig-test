from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.policy_researcher import PolicyResearchAgent
from src.config import Settings


class DummyLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.invocations = []

    def invoke(self, messages, response_format=None):  # noqa: D401 - mimic ChatGroq interface
        self.invocations.append(
            {
                "messages": messages,
                "response_format": response_format,
            }
        )
        return SimpleNamespace(content=self.reply)


@pytest.fixture()
def taxonomy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Taxonomy" / "Taxonomy_Hackathon.json"


def test_policy_research_agent_parses_llm_payload(taxonomy_path: Path) -> None:
    response_payload = {
        "products": [
            {
                "product": "Product A",
                "tier": "Gold",
                "benefits": [
                    {
                        "name": "medical",
                        "why_eligible": "Matches taxonomy criteria",
                        "parameters": None,
                        "conditions": [],
                    }
                ],
            }
        ],
        "reasoning": "Sample reasoning",
    }
    dummy_llm = DummyLLM(json.dumps(response_payload))

    agent = PolicyResearchAgent(
        settings=Settings(
            groq_api_key="test-key",
            groq_model="test-model",
            taxonomy_path=taxonomy_path,
        ),
        llm=dummy_llm,
    )

    result = agent.run(
        user_query="I want to go to Japan",
        recommended_products=["Product A"],
        tiers=["Gold"],
        chat_history=[("user", "I want to go to Japan")],
    )

    assert result.products == response_payload["products"]
    assert result.reasoning == "Sample reasoning"
    assert dummy_llm.invocations, "LLM should have been invoked"
    assert dummy_llm.invocations[0]["response_format"] == {"type": "json_object"}


def test_policy_research_agent_handles_missing_products(taxonomy_path: Path) -> None:
    dummy_llm = DummyLLM("{}")
    agent = PolicyResearchAgent(
        settings=Settings(
            groq_api_key="test-key",
            groq_model="test-model",
            taxonomy_path=taxonomy_path,
        ),
        llm=dummy_llm,
    )

    result = agent.run(
        user_query="No recommendation yet",
        recommended_products=[],
        tiers=[],
        chat_history=[],
    )

    assert result.products == []
    assert not dummy_llm.invocations, "LLM should not be called when no products are provided"


def test_policy_research_agent_reload_taxonomy_when_file_changes(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    initial_payload = {
        "layers": {
            "layer_1_general_conditions": [],
            "layer_2_benefits": [],
            "layer_3_benefit_conditions": [],
        }
    }
    taxonomy_path.write_text(json.dumps(initial_payload), encoding="utf-8")

    dummy_llm = DummyLLM(json.dumps({"products": [], "reasoning": ""}))
    agent = PolicyResearchAgent(
        settings=Settings(
            groq_api_key="test-key",
            groq_model="test-model",
            taxonomy_path=taxonomy_path,
        ),
        llm=dummy_llm,
    )

    agent.run(
        user_query="Initial run",
        recommended_products=["Plan"],
        tiers=["Standard"],
        chat_history=[],
    )

    dummy_llm.invocations.clear()

    updated_payload = {
        "layers": {
            "layer_1_general_conditions": [],
            "layer_2_benefits": [
                {
                    "benefit_name": "new_benefit",
                    "parameters": [],
                    "products": {
                        "Plan": {
                            "condition_exist": True,
                            "parameters": {
                                "coverage_limit": "$500",
                                "sub_limits": {},
                            },
                        }
                    },
                }
            ],
            "layer_3_benefit_conditions": [],
        }
    }
    taxonomy_path.write_text(json.dumps(updated_payload), encoding="utf-8")
    new_timestamp = taxonomy_path.stat().st_mtime + 5
    os.utime(taxonomy_path, (new_timestamp, new_timestamp))

    agent.run(
        user_query="Need updated context",
        recommended_products=["Plan"],
        tiers=["Standard"],
        chat_history=[],
    )

    assert dummy_llm.invocations, "LLM should have been invoked after taxonomy update"
    llm_prompt = dummy_llm.invocations[0]["messages"][1].content
    assert "new_benefit" in llm_prompt
    assert "$500" in llm_prompt
