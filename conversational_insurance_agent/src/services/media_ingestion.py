from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import httpx
import pdfplumber
from groq import Groq

from ..config import Settings, get_settings
from ..utils.logging import logger


@dataclass
class MediaAttachment:
    url: str
    content_type: str
    filename: Optional[str] = None
    media_sid: Optional[str] = None


class GroqMediaIngestor:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)

    async def analyse(self, attachments: List[MediaAttachment]) -> List[str]:
        if not attachments:
            return []

        tasks = [self._analyse_single(attachment) for attachment in attachments]
        results: List[str] = []
        for attachment, task in zip(attachments, tasks):
            try:
                results.append(await task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "groq_media.analysis_failed",
                    media_sid=attachment.media_sid,
                    content_type=attachment.content_type,
                    error=str(exc),
                )
        return results

    async def _analyse_single(self, attachment: MediaAttachment) -> str:
        data = await self._download(attachment.url)
        if attachment.content_type.startswith("image/"):
            return await self._describe_image(data, attachment.content_type)
        if attachment.content_type.lower() == "application/pdf":
            return await self._summarise_pdf(data, attachment.filename)
        logger.warning(
            "groq_media.unsupported_type",
            content_type=attachment.content_type,
            media_sid=attachment.media_sid,
        )
        return f"Received unsupported media type {attachment.content_type}."

    async def _download(self, url: str) -> bytes:
        auth: Optional[httpx.Auth] = None
        if self._settings.twilio_account_sid and self._settings.twilio_auth_token:
            auth = httpx.BasicAuth(
                self._settings.twilio_account_sid,
                self._settings.twilio_auth_token,
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, auth=auth)
            response.raise_for_status()
            return response.content

    async def _describe_image(self, data: bytes, content_type: str) -> str:
        encoded = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{content_type};base64,{encoded}"

        def _call() -> str:
            completion = self._client.chat.completions.create(
                model=self._settings.groq_vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that analyses user provided media to "
                            "extract concise, factual details relevant to travel insurance."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Provide a short bullet list (max 4 bullets) summarising the "
                                    "key information visible in this image that could support a "
                                    "travel insurance enquiry or claim."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    },
                ],
                temperature=0.2,
                max_completion_tokens=500,
            )
            message = completion.choices[0].message
            return message.content.strip() if message and message.content else ""

        return await asyncio.to_thread(_call)

    async def _summarise_pdf(self, data: bytes, filename: Optional[str]) -> str:
        def _extract_text() -> str:
            text_parts: List[str] = []
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
                    if len("\n".join(text_parts)) > 8000:
                        break
            return "\n".join(text_parts)

        raw_text = await asyncio.to_thread(_extract_text)
        truncated = raw_text[:6000]
        prompt = (
            "Summarise the key facts from this travel-related document. Focus on "
            "dates, destinations, travellers, costs, incidents, or coverage details "
            "relevant for insurance support. Present the summary as bullet points."
        )

        def _call() -> str:
            completion = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You condense uploaded travel documents into actionable insights "
                            "for an insurance concierge."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nDocument name: {filename or 'uploaded document'}\n\n{truncated}",
                    },
                ],
                temperature=0.2,
                max_completion_tokens=500,
            )
            message = completion.choices[0].message
            return message.content.strip() if message and message.content else ""

        return await asyncio.to_thread(_call)
