from __future__ import annotations

from typing import Any, Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .core.orchestrator import ConversationalOrchestrator
from .core.setup import build_orchestrator
from .utils.logging import configure_logging, logger


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable identifier for the conversation thread")
    message: str = Field(..., description="User utterance")
    channel: str = Field("web", description="Channel identifier such as web, whatsapp, telegram")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    refresh: bool = False


class WhatsAppWebhook(BaseModel):
    from_number: str = Field(..., alias="from")
    body: str
    wa_id: Optional[str] = Field(None, alias="waId")


class TelegramWebhook(BaseModel):
    chat_id: str
    text: str
    username: Optional[str] = None


configure_logging()
app = FastAPI(title="Ancileo Conversational Insurance Platform", version="0.1.0")
_orchestrator: ConversationalOrchestrator | None = None


def get_orchestrator() -> ConversationalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


def get_config() -> Settings:
    return get_settings()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    orchestrator: ConversationalOrchestrator = Depends(get_orchestrator),
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

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


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    payload: WhatsAppWebhook,
    orchestrator: ConversationalOrchestrator = Depends(get_orchestrator),
):
    response = await orchestrator.handle_message(
        session_id=payload.wa_id or payload.from_number,
        user_message=payload.body,
        channel="whatsapp",
    )
    return {"reply": response["reply"]}


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
    return {"reply": response["reply"]}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8080, reload=False)
