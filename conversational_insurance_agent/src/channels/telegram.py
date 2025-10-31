from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TelegramMessage:
    chat_id: str
    text: str
    username: Optional[str] = None
    raw: Optional[Dict[str, str]] = None

    @classmethod
    def from_bot_update(cls, update: Dict[str, Any]) -> "TelegramMessage":
        message = update.get("message", {})
        chat = message.get("chat", {})
        return cls(
            chat_id=str(chat.get("id")),
            text=message.get("text", ""),
            username=chat.get("username"),
            raw=update,
        )
