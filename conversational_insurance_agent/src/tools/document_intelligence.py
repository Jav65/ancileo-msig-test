from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

from ..utils.logging import logger


DATE_PATTERNS = [
    r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]


class DocumentIntelligenceTool:
    def parse_trip_document(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(path)
        else:
            raise ValueError("Only PDF documents are supported in this prototype.")

        dates = self._extract_dates(text)
        destinations = self._extract_destinations(text)
        passengers = self._extract_passenger_names(text)
        budget = self._estimate_trip_cost(text)

        logger.info(
            "doc_intel.parsed",
            file=path.name,
            dates=len(dates),
            destinations=len(destinations),
            passengers=len(passengers),
        )

        return {
            "file": path.name,
            "dates": dates,
            "destinations": destinations,
            "passengers": passengers,
            "estimated_trip_cost": budget,
            "raw_preview": text[:1000],
        }

    def _extract_pdf_text(self, path: Path) -> str:
        contents: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                contents.append(page.extract_text() or "")
        return "\n".join(contents)

    def _extract_dates(self, text: str) -> List[str]:
        matches: List[str] = []
        for pattern in DATE_PATTERNS:
            matches.extend(re.findall(pattern, text))

        parsed = []
        for item in matches:
            try:
                parsed_date = self._parse_date(item)
                if parsed_date:
                    parsed.append(parsed_date.strftime("%Y-%m-%d"))
            except ValueError:
                continue
        return sorted(set(parsed))

    def _parse_date(self, value: str) -> Optional[datetime]:
        for fmt in ["%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _extract_destinations(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines()]
        airports = []
        for line in lines:
            if "depart" in line.lower() or "arrive" in line.lower():
                airports.append(line)
        # fallback: look for uppercase place names
        uppercase_words = re.findall(r"\b([A-Z]{3,})\b", text)
        airports.extend(uppercase_words)
        return list(dict.fromkeys(airports))[:10]

    def _extract_passenger_names(self, text: str) -> List[str]:
        pattern = re.compile(r"Passenger\s*[:\-]\s*(.+)")
        names = pattern.findall(text)
        cleaned = []
        for name in names:
            tokens = re.split(r",|/|\n", name)
            cleaned.extend(token.strip() for token in tokens if token.strip())
        return list(dict.fromkeys(cleaned))[:6]

    def _estimate_trip_cost(self, text: str) -> Optional[float]:
        money_pattern = re.compile(r"(?:USD|SGD|S\$|US\$)?\s*([0-9]{2,}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")
        matches = [float(value.replace(",", "")) for value in money_pattern.findall(text)]
        if not matches:
            return None
        return max(matches)
