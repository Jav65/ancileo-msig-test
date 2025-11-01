from __future__ import annotations

from typing import Any, Dict, List, Optional

import html

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

try:  # pragma: no cover - pydantic v2 optional
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 fallback
    ConfigDict = None  # type: ignore[misc, assignment]

from .channels.whatsapp import WhatsAppMessage
from .config import Settings, get_settings
from .core.orchestrator import ConversationalOrchestrator
from .core.setup import build_orchestrator
from .services.media_ingestion import GroqMediaIngestor, MediaAttachment
from .state.client_context import ClientDatum
from .utils.logging import configure_logging, logger


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable identifier for the conversation thread")
    message: str = Field(..., description="User utterance")
    channel: str = Field("web", description="Channel identifier such as web, whatsapp, telegram")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    clients: List[ClientDatum] = Field(
        default_factory=list,
        alias="clientData",
        description="Known traveller details supplied by the integrating partner",
    )

    if ConfigDict:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            allow_population_by_field_name = True

class ToolRun(BaseModel):
    name: str
    input: Dict[str, Any]
    result: Any
    tool_call_id: Optional[str] = None


class ActionRequest(BaseModel):
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class ChatResponse(BaseModel):
    output: str
    actions: List[ActionRequest] = Field(default_factory=list)
    tool_used: Optional[str] = None
    tool_result: Optional[Any] = None
    tool_runs: List[ToolRun] = Field(default_factory=list)


class IngestRequest(BaseModel):
    refresh: bool = False


class TelegramWebhook(BaseModel):
    chat_id: str
    text: str
    username: Optional[str] = None


configure_logging()
app = FastAPI(title="Ancileo Conversational Insurance Platform", version="0.1.0")
_orchestrator: ConversationalOrchestrator | None = None
_media_ingestor: GroqMediaIngestor | None = None


def get_orchestrator() -> ConversationalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


def get_config() -> Settings:
    return get_settings()


def get_media_ingestor() -> GroqMediaIngestor:
    global _media_ingestor
    if _media_ingestor is None:
        _media_ingestor = GroqMediaIngestor(get_settings())
    return _media_ingestor


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    orchestrator: ConversationalOrchestrator = Depends(get_orchestrator),
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if payload.clients:
        orchestrator.merge_clients(
            session_id=payload.session_id,
            clients=payload.clients,
            source=payload.channel,
        )

    response = await orchestrator.handle_message(
        session_id=payload.session_id,
        user_message=payload.message,
        channel=payload.channel,
    )

    return ChatResponse(**response)


@app.post("/tools/policy/index")
async def rebuild_policy_index(
    request: IngestRequest,
    background: BackgroundTasks,
):
    orchestrator = get_orchestrator()
    policy_tool = next(
        (tool for tool in orchestrator._tool_map.values() if tool.name == "policy_lookup"),
        None,
    )
    if not policy_tool:
        raise HTTPException(status_code=500, detail="Policy tool not available")

    def _task() -> None:
        logger.info("policy_index.rebuild.start")
        handler = policy_tool.handler
        policy_instance = getattr(handler, "__self__", None)
        if policy_instance is None:
            raise RuntimeError("Policy tool handler is not bound to an instance")
        result = policy_instance.ingest(refresh=request.refresh)
        logger.info("policy_index.rebuild.finish", result=result)

    background.add_task(_task)
    return JSONResponse({"status": "queued"})


@app.get("/healthz")
async def healthcheck(settings: Settings = Depends(get_config)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "groq_model": settings.groq_model,
        "payments_base_url": settings.payments_base_url,
    }


@app.post("/webhooks/whatsapp", response_class=Response)
async def whatsapp_webhook(
    request: Request,
    orchestrator: ConversationalOrchestrator = Depends(get_orchestrator),
    media_ingestor: GroqMediaIngestor = Depends(get_media_ingestor),
):
    form_data = await request.form()
    payload: Dict[str, str] = {key: value for key, value in form_data.items()}
    message = WhatsAppMessage.from_twilio_payload(payload)

    metadata = message.metadata or {}
    profile_name = metadata.get("ProfileName") or metadata.get("profile_name")
    whatsapp_client = ClientDatum(
        client_id=message.wa_id or message.sender,
        source="whatsapp",
        personal_info={
            "name": profile_name,
            "phone_number": message.sender,
        },
        extra={"whatsapp": {"metadata": metadata}},
    )
    orchestrator.merge_clients(message.session_id, [whatsapp_client], source="whatsapp")

    logger.info(
        "whatsapp_webhook.received",
        sender=message.sender,
        wa_id=message.wa_id,
        attachments=len(message.attachments),
    )

    media_attachments = [
        MediaAttachment(
            url=attachment.url,
            content_type=attachment.content_type,
            filename=attachment.filename,
            media_sid=attachment.media_sid,
        )
        for attachment in message.attachments
    ]

    attachment_summaries = await media_ingestor.analyse(media_attachments)

    user_message = message.text.strip()

    if attachment_summaries:
        insights = []
        for index, summary in enumerate(attachment_summaries, start=1):
            label = f"Attachment {index}"
            summary_text = summary or "No insight available."
            insights.append(f"[{label}] {summary_text}")

        insights_block = "\n\n".join(insights)
        if user_message:
            user_message = f"{user_message}\n\n[Attachment Insights]\n{insights_block}"
        else:
            user_message = f"[Attachment Insights]\n{insights_block}"

    if not user_message:
        user_message = "User sent media with no accompanying text."

    response = await orchestrator.handle_message(
        session_id=message.session_id,
        user_message=user_message,
        channel="whatsapp",
    )

    reply_text = response.get("output", "")
    twiml = _render_twiml(reply_text)
    return Response(content=twiml, media_type="application/xml")


def _render_twiml(message: str) -> str:
    escaped = html.escape(message or "")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        f"<Response><Message>{escaped}</Message></Response>"
    )


@app.post("/webhooks/telegram")
async def telegram_webhook(
    payload: TelegramWebhook,
    orchestrator: ConversationalOrchestrator = Depends(get_orchestrator),
):
    response = await orchestrator.handle_message(
        session_id=payload.chat_id,
        user_message=payload.text,
        channel="telegram",
    )
    return {"output": response.get("output", "")}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=False)
