from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WhatsAppMediaAttachment:
    url: str
    content_type: str
    filename: Optional[str] = None
    media_sid: Optional[str] = None

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")

    @property
    def is_pdf(self) -> bool:
        return self.content_type.lower() == "application/pdf"


@dataclass
class WhatsAppMessage:
    sender: str
    text: str
    wa_id: str | None = None
    metadata: Dict[str, str] | None = None
    attachments: List[WhatsAppMediaAttachment] = field(default_factory=list)

    @classmethod
    def from_twilio_payload(cls, payload: Dict[str, str]) -> "WhatsAppMessage":
        attachments: List[WhatsAppMediaAttachment] = []
        try:
            num_media = int(payload.get("NumMedia", "0") or 0)
        except (TypeError, ValueError):
            num_media = 0

        for index in range(num_media):
            url = payload.get(f"MediaUrl{index}")
            content_type = payload.get(f"MediaContentType{index}") or ""
            if not url:
                continue
            attachments.append(
                WhatsAppMediaAttachment(
                    url=url,
                    content_type=content_type,
                    filename=payload.get(f"MediaFilename{index}"),
                    media_sid=payload.get(f"MediaSid{index}"),
                )
            )

        return cls(
            sender=payload.get("From", ""),
            text=payload.get("Body", ""),
            wa_id=payload.get("WaId"),
            metadata={
                k: v
                for k, v in payload.items()
                if k not in {"From", "Body", "WaId"} and not k.startswith("Media")
            },
            attachments=attachments,
        )

    @property
    def session_id(self) -> str:
        return self.wa_id or self.sender
