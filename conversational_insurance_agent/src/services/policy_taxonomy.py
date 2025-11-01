from __future__ import annotations

"""Utilities for extracting structured taxonomy layers from policy PDFs."""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pdfplumber
from groq import Groq
from jsonschema import Draft7Validator, ValidationError

from ..config import Settings, get_settings
from ..utils.logging import logger


@dataclass
class IngestCfg:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_chars: int = 7000
    take_chunks: int = 8
    temperature: float = 0.1
    retries: int = 3
    max_tokens: int = 2200

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "IngestCfg":
        settings = settings or get_settings()
        cfg = cls()
        cfg.groq_api_key = settings.groq_api_key
        cfg.groq_model = settings.groq_model
        return cfg


L1_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "condition": {"type": "string"},
            "condition_type": {"type": "string", "enum": ["eligibility", "exclusion"]},
            "products": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "condition_exist": {"type": "boolean"},
                        "original_text": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["condition_exist", "original_text", "parameters"],
                },
            },
        },
        "required": ["condition", "condition_type", "products"],
    },
}


L2_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "benefit_name": {"type": "string"},
            "parameters": {"type": "array"},
            "products": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "condition_exist": {"type": "boolean"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "coverage_limit": {},
                                "sub_limits": {"type": "object"},
                            },
                            "required": ["coverage_limit", "sub_limits"],
                        },
                    },
                    "required": ["condition_exist", "parameters"],
                },
            },
        },
        "required": ["benefit_name", "products"],
    },
}


L3_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "benefit_name": {"type": "string"},
            "condition": {"type": "string"},
            "condition_type": {
                "type": "string",
                "enum": ["benefit_eligibility", "benefit_exclusion"],
            },
            "parameters": {"type": "array"},
            "products": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "condition_exist": {"type": "boolean"},
                        "original_text": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["condition_exist", "original_text", "parameters"],
                },
            },
        },
        "required": ["benefit_name", "condition", "condition_type", "products"],
    },
}


def _parse_strict_json(text: str) -> Any:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S)
    match = re.search(r"[\[\{]", content)
    if match:
        content = content[match.start() :]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        lines = content.splitlines()
        for cut in range(len(lines), 0, -1):
            try:
                return json.loads("\n".join(lines[:cut]))
            except json.JSONDecodeError:
                continue
        raise


class PolicyIngestor:
    def __init__(self, cfg: IngestCfg | None = None, client: Groq | None = None) -> None:
        self.cfg = cfg or IngestCfg()
        if not self.cfg.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = client or Groq(api_key=self.cfg.groq_api_key)

    def _load_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                pages.append((index, page.extract_text() or ""))
        return pages

    def _chunk(self, pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        buffer = ""
        start_page: int | None = None
        end_page: int | None = None
        for page_number, text in pages:
            clean = (text or "").strip()
            if not clean:
                continue
            if start_page is None:
                start_page = page_number
            if len(buffer) + len(clean) + 16 <= self.cfg.max_chars:
                buffer += f"\n\n[PAGE {page_number}]\n{clean}"
                end_page = page_number
            else:
                if buffer.strip():
                    chunks.append(
                        {
                            "pages": (start_page, end_page or start_page),
                            "text": buffer.strip(),
                        }
                    )
                buffer = f"[PAGE {page_number}]\n{clean}"
                start_page = page_number
                end_page = page_number
        if buffer.strip():
            chunks.append({"pages": (start_page, end_page or start_page), "text": buffer.strip()})
        return chunks

    def _ask_json(self, system_prompt: str, user_prompt: str) -> Any:
        error: Exception | None = None
        for attempt in range(self.cfg.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                message = response.choices[0].message
                return _parse_strict_json(message.content if message and message.content else "")
            except Exception as exc:  # pragma: no cover - defensive against API errors
                error = exc
                logger.warning(
                    "policy_taxonomy.llm_retry",
                    attempt=attempt + 1,
                    retries=self.cfg.retries,
                    error=str(exc),
                )
                time.sleep(1.2 * (attempt + 1))
        raise RuntimeError(f"Groq call failed: {error}")

    def _sys_l1(self) -> str:
        return (
            "Extract Layer 1: General Conditions from travel-insurance policy text.\n"
            "Return STRICT JSON ONLY (no prose or code fences).\n"
            "Each item MUST have:\n"
            '{\n'
            '  "condition": "<snake_case_name>",\n'
            '  "condition_type": "eligibility" | "exclusion",\n'
            '  "products": {\n'
            '    "<product_label>": {\n'
            '      "condition_exist": true/false,\n'
            '      "original_text": "exact policy wording",\n'
            '      "parameters": {\n'
            '        // key-value pairs capturing explicit criteria or thresholds found in the text\n'
            '        // e.g., {"age_limit": "12 years", "requires_parent_guardian": true, "territory": "worldwide"}\n'
            '      }\n'
            '    }\n'
            '  }\n'
            '}\n'
            "Do not leave 'parameters' empty unless no measurable or categorical detail exists. "
            "Infer structured keys such as age, duration, distance, monetary thresholds, frequency, or role requirements."
            "Your output MUST conform exactly to the JSON shape described ? do not change field names or omit fields. "
            "Only vary the contents of the 'parameters' object as key-value pairs."
        )

    def _user_l1(self, product_label: str, chunks: List[Dict[str, Any]]) -> str:
        shape = (
            '[{"condition":"<snake_case_key>",'  # noqa: E501
            '"condition_type":"eligibility|exclusion",'  # noqa: E501
            '"products":{"'
            + product_label
            + '":{'  # noqa: E501
            '"condition_exist":true,'  # noqa: E501
            '"original_text":"...",'  # noqa: E501
            '"parameters":{"<parameter_key>":"<parameter_value>", "...":"..."}'  # noqa: E501
            '}}}]'
        )
        text = "\n\n".join(
            [f"[PAGES {start}-{end}]\n{chunk['text']}" for (start, end), chunk in [(item["pages"], item) for item in chunks]]
        )
        return (
            f"product_label: {product_label}\n"
            f"Return an array in EXACT shape:\n{shape}\n\n"
            "Policy text:\n-----\n"
            f"{text}"
        )

    def _sys_l2(self) -> str:
        return (
            "Extract Layer 2: ALL coverage benefits and their limits/qualifiers from the policy text.\n"
            "Return STRICT JSON ONLY (no prose, no code fences).\n"
            "Each array item MUST be exactly:\n"
            '{\n'
            '  "benefit_name": "<snake_case_identifier>",\n'
            '  "parameters": [],\n'
            '  "products": { "<product_label>": {\n'
            '      "condition_exist": true/false,\n'
            '      "parameters": {\n'
            '        "coverage_limit": "<amount or descriptor if present>",\n'
            '        "sub_limits": { /* zero or more key-value pairs that appear in the text */ }\n'
            '      }\n'
            '  }}\n'
            '}\n'
            "Rules:\n"
            "- Discover EVERY benefit section present (e.g., medical expenses, trip cancellation, delay, baggage, personal liability,\n"
            "  evacuation, adventurous activities, rental vehicle excess, etc.).\n"
            "- Use short, machine-friendly snake_case for benefit_name (e.g., trip_cancellation, delayed_baggage, personal_liability).\n"
            "- If a coverage cap exists, put it in coverage_limit (keep units/format from text); if none is stated, omit that field or set a descriptive value.\n"
            "- Put any nested caps/thresholds/waiting_periods/deductibles/percentages into sub_limits as key-value pairs (only if present in the text).\n"
            "- Do NOT invent keys. Use exact facts. If a field isn?t present in the text, leave it out.\n"
            "- Your output MUST conform exactly to the JSON shape above: only vary the contents of the 'parameters' object."
        )

    def _user_l2(self, product_label: str, chunks: List[Dict[str, Any]]) -> str:
        shape = (
            '[{"benefit_name":"<snake_case_identifier>",'  # noqa: E501
            '"parameters":[],'
            '"products":{"'
            + product_label
            + '":{'  # noqa: E501
            '"condition_exist":true,'  # noqa: E501
            '"parameters":{"coverage_limit":"<value_or_descriptor_if_present>",'  # noqa: E501
            '"sub_limits":{"<key>":"<value>"}}'  # noqa: E501
            '}}}]'
        )
        text = "\n\n".join(
            [f"[PAGES {start}-{end}]\n{chunk['text']}" for (start, end), chunk in [(item["pages"], item) for item in chunks]]
        )
        return (
            f"product_label: {product_label}\n"
            f"Return array in EXACT shape:\n{shape}\n\n"
            "Policy text:\n-----\n"
            f"{text}"
        )

    def _sys_l3(self) -> str:
        return (
            "Extract Layer 3: Benefit-specific conditions (eligibilities/exclusions) tied to EACH benefit.\n"
            "Return STRICT JSON ONLY (no prose, no code fences).\n"
            "Each array item MUST be exactly:\n"
            '{\n'
            '  "benefit_name": "<parent_benefit_snake_case>",\n'
            '  "condition": "<specific_condition_snake_case>",\n'
            '  "condition_type": "benefit_eligibility" | "benefit_exclusion",\n'
            '  "parameters": [],\n'
            '  "products": { "<product_label>": {\n'
            '      "condition_exist": true/false,\n'
            '      "original_text": "minimal exact quote from the policy",\n'
            '      "parameters": { /* zero or more key-value pairs present in the text */ }\n'
            '  }}\n'
            '}\n'
            "Rules:\n"
            "- For EACH benefit discovered in Layer 2, find its explicit conditions (e.g., time limits, documentation requirements,\n"
            "  minimum thresholds, known-circumstance exclusions, activity/location qualifiers, age rules, waiting periods, deductibles, etc.).\n"
            "- Use short snake_case names for both benefit_name and condition, reflecting the policy?s wording.\n"
            "- In 'products.<product_label>.parameters', include ONLY measurable/categorical keys that appear in the text\n"
            "  (e.g., {\"time_limit\":\"90 days\", \"minimum_amount\":\"$500\", \"requires_doctor_note\":true}).\n"
            "- Do NOT invent keys. Only include parameters present in the text. Preserve units/format.\n"
            "- Your output MUST conform exactly to the JSON shape above; only vary the contents of the 'parameters' object."
        )

    def _user_l3(self, product_label: str, chunks: List[Dict[str, Any]]) -> str:
        shape = (
            '[{"benefit_name":"<parent_benefit>",'  # noqa: E501
            '"condition":"<specific_condition>",'  # noqa: E501
            '"condition_type":"benefit_eligibility|benefit_exclusion",'  # noqa: E501
            '"parameters":[],'  # noqa: E501
            '"products":{"'
            + product_label
            + '":{'  # noqa: E501
            '"condition_exist":true,'  # noqa: E501
            '"original_text":"...",'  # noqa: E501
            '"parameters":{"<parameter_key>":"<parameter_value>"}'  # noqa: E501
            '}}}]'
        )
        text = "\n\n".join(
            [f"[PAGES {start}-{end}]\n{chunk['text']}" for (start, end), chunk in [(item["pages"], item) for item in chunks]]
        )
        return (
            f"product_label: {product_label}\n"
            f"Return array in EXACT shape:\n{shape}\n\n"
            "Policy text:\n-----\n"
            f"{text}"
        )

    def run_layer1(self, pdf_path: str, product_label: str) -> Dict[str, Any]:
        pages = self._load_pages(pdf_path)
        chunks = self._chunk(pages)[: self.cfg.take_chunks]
        raw = self._ask_json(self._sys_l1(), self._user_l1(product_label, chunks))
        try:
            Draft7Validator(L1_SCHEMA).validate(raw)
        except ValidationError as exc:
            raise ValueError(f"L1 invalid: {exc.message}") from exc
        return {"layer_1": raw}

    def run_layer2(self, pdf_path: str, product_label: str) -> Dict[str, Any]:
        pages = self._load_pages(pdf_path)
        chunks = self._chunk(pages)[: self.cfg.take_chunks]
        raw = self._ask_json(self._sys_l2(), self._user_l2(product_label, chunks))
        try:
            Draft7Validator(L2_SCHEMA).validate(raw)
        except ValidationError as exc:
            raise ValueError(f"L2 invalid: {exc.message}") from exc
        return {"layer_2": raw}

    def run_layer3(self, pdf_path: str, product_label: str) -> Dict[str, Any]:
        pages = self._load_pages(pdf_path)
        chunks = self._chunk(pages)[: self.cfg.take_chunks]
        raw = self._ask_json(self._sys_l3(), self._user_l3(product_label, chunks))
        try:
            Draft7Validator(L3_SCHEMA).validate(raw)
        except ValidationError as exc:
            raise ValueError(f"L3 invalid: {exc.message}") from exc
        return {"layer_3": raw}


def extract_all_layers(
    pdf_path: str,
    product_label: str,
    ingestor: PolicyIngestor | None = None,
) -> Dict[str, Any]:
    if not product_label or not product_label.strip():
        raise ValueError("product_label cannot be empty")
    ing = ingestor or PolicyIngestor()
    layer1 = ing.run_layer1(pdf_path, product_label)["layer_1"]
    layer2 = ing.run_layer2(pdf_path, product_label)["layer_2"]
    layer3 = ing.run_layer3(pdf_path, product_label)["layer_3"]
    return {
        "layer_1_general_conditions": layer1,
        "layer_2_benefits": layer2,
        "layer_3_benefit_conditions": layer3,
    }


__all__ = [
    "IngestCfg",
    "PolicyIngestor",
    "extract_all_layers",
    "L1_SCHEMA",
    "L2_SCHEMA",
    "L3_SCHEMA",
]
