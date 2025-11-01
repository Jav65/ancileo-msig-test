from __future__ import annotations

from io import BytesIO
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from src.main import app, get_policy_ingestor


class StubPolicyIngestor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, str]] = []

    def _products(self, product_label: str) -> Dict[str, Dict[str, object]]:
        return {
            product_label: {
                "condition_exist": True,
                "original_text": "Example original policy text.",
                "parameters": {"example_parameter": "value"},
            }
        }

    def run_layer1(self, pdf_path: str, product_label: str) -> Dict[str, object]:
        self.calls.append({"layer": "layer1", "label": product_label, "path": pdf_path})
        return {
            "layer_1": [
                {
                    "condition": "example_condition",
                    "condition_type": "eligibility",
                    "products": self._products(product_label),
                }
            ]
        }

    def run_layer2(self, pdf_path: str, product_label: str) -> Dict[str, object]:
        self.calls.append({"layer": "layer2", "label": product_label, "path": pdf_path})
        return {
            "layer_2": [
                {
                    "benefit_name": "example_benefit",
                    "parameters": [],
                    "products": {
                        product_label: {
                            "condition_exist": True,
                            "parameters": {
                                "coverage_limit": "$1,000",
                                "sub_limits": {},
                            },
                        }
                    },
                }
            ]
        }

    def run_layer3(self, pdf_path: str, product_label: str) -> Dict[str, object]:
        self.calls.append({"layer": "layer3", "label": product_label, "path": pdf_path})
        return {
            "layer_3": [
                {
                    "benefit_name": "example_benefit",
                    "condition": "example_condition",
                    "condition_type": "benefit_eligibility",
                    "parameters": [],
                    "products": self._products(product_label),
                }
            ]
        }


@pytest.fixture(autouse=True)
def groq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def stub_ingestor() -> StubPolicyIngestor:
    stub = StubPolicyIngestor()
    app.dependency_overrides[get_policy_ingestor] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_policy_ingestor, None)


@pytest.fixture()
def sample_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extract_taxonomy_success(client: TestClient, stub_ingestor: StubPolicyIngestor, sample_pdf: bytes) -> None:
    response = client.post(
        "/taxonomy/extract",
        data={"product_label": "demo_plan"},
        files={"pdf": ("policy.pdf", sample_pdf, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "layer_1_general_conditions": [
            {
                "condition": "example_condition",
                "condition_type": "eligibility",
                "products": {
                    "demo_plan": {
                        "condition_exist": True,
                        "original_text": "Example original policy text.",
                        "parameters": {"example_parameter": "value"},
                    }
                },
            }
        ],
        "layer_2_benefits": [
            {
                "benefit_name": "example_benefit",
                "parameters": [],
                "products": {
                    "demo_plan": {
                        "condition_exist": True,
                        "parameters": {
                            "coverage_limit": "$1,000",
                            "sub_limits": {},
                        },
                    }
                },
            }
        ],
        "layer_3_benefit_conditions": [
            {
                "benefit_name": "example_benefit",
                "condition": "example_condition",
                "condition_type": "benefit_eligibility",
                "parameters": [],
                "products": {
                    "demo_plan": {
                        "condition_exist": True,
                        "original_text": "Example original policy text.",
                        "parameters": {"example_parameter": "value"},
                    }
                },
            }
        ],
    }
    assert {call["layer"] for call in stub_ingestor.calls} == {"layer1", "layer2", "layer3"}


def test_extract_taxonomy_rejects_non_pdf(client: TestClient, stub_ingestor: StubPolicyIngestor) -> None:
    response = client.post(
        "/taxonomy/extract",
        data={"product_label": "demo_plan"},
        files={"pdf": ("policy.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only PDF policy documents are supported"
