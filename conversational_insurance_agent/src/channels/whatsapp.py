from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class WhatsAppMessage:
    sender: str
    text: str
    wa_id: str | None = None
    metadata: Dict[str, str] | None = None

    @classmethod
    def from_twilio_payload(cls, payload: Dict[str, str]) -> "WhatsAppMessage":
        return cls(
            sender=payload.get("From", ""),
            text=payload.get("Body", ""),
            wa_id=payload.get("WaId"),
            metadata={k: v for k, v in payload.items() if k not in {"From", "Body", "WaId"}},
        )

    @property
    def session_id(self) -> str:
        return self.wa_id or self.sender
