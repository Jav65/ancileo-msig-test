from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..core.orchestrator import ConversationalOrchestrator
from ..utils.logging import logger
from .mock_db import MockUserRecord, authenticate_user, get_user


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


router = APIRouter(prefix="/integration", tags=["integration"])

SESSION_USER_KEY = "integration_user"
SESSION_ID_KEY = "integration_session_id"
SESSION_CHANNEL = "integration_channel"
DEFAULT_CHANNEL = "integration_web"


class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, description="User utterance")


def _ensure_session_id(request: Request, user: MockUserRecord) -> str:
    session_id = request.session.get(SESSION_ID_KEY)
    if session_id:
        return session_id
    session_id = user.session_id
    request.session[SESSION_ID_KEY] = session_id
    return session_id


def _get_current_user(request: Request) -> MockUserRecord:
    username = request.session.get(SESSION_USER_KEY)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    record = get_user(username)
    if not record:
        request.session.pop(SESSION_USER_KEY, None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User session expired")
    return record


def mount_integration_static(app) -> None:  # pragma: no cover - runtime wiring
    from fastapi.staticfiles import StaticFiles

    app.mount("/integration/static", StaticFiles(directory=str(STATIC_DIR)), name="integration-static")


def get_portal_orchestrator(request: Request) -> ConversationalOrchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        from ..core.setup import build_orchestrator

        orchestrator = build_orchestrator()
        request.app.state.orchestrator = orchestrator
    return orchestrator


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if request.session.get(SESSION_USER_KEY):
        return RedirectResponse(url="/integration/chat", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
async def login_submit(
    request: Request,
    orchestrator: ConversationalOrchestrator = Depends(get_portal_orchestrator),
):
    form = await request.form()
    username = (form.get("username") or "").strip().lower()
    password = (form.get("password") or "").strip()

    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Please provide both email and password."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    record = authenticate_user(username, password)
    if not record:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials. Try alice@example.com / travel123."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.session[SESSION_USER_KEY] = record.username
    request.session[SESSION_CHANNEL] = DEFAULT_CHANNEL
    session_id = _ensure_session_id(request, record)

    client = record.build_client()
    orchestrator.merge_clients(session_id=session_id, clients=[client], source=DEFAULT_CHANNEL)

    logger.info("integration.login", username=record.username, session_id=session_id)
    return RedirectResponse(url="/integration/chat", status_code=status.HTTP_302_FOUND)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    try:
        user = _get_current_user(request)
    except HTTPException:
        return RedirectResponse(url="/integration/login", status_code=status.HTTP_302_FOUND)

    client = user.build_client()
    context = {
        "request": request,
        "user": user,
        "client": client,
        "primary_trip": client.trips[0] if client.trips else None,
        "past_policies": user.past_policies,
    }
    return templates.TemplateResponse("chat.html", context)


@router.post("/chat/send")
async def chat_send(
    payload: ChatPayload,
    request: Request,
    orchestrator: ConversationalOrchestrator = Depends(get_portal_orchestrator),
) -> JSONResponse:
    user = _get_current_user(request)
    session_id = _ensure_session_id(request, user)
    channel = request.session.get(SESSION_CHANNEL, DEFAULT_CHANNEL)

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    client = user.build_client()
    orchestrator.merge_clients(session_id=session_id, clients=[client], source=channel)

    logger.info("integration.chat_send", username=user.username, session_id=session_id)

    response = await orchestrator.handle_message(
        session_id=session_id,
        user_message=message,
        channel=channel,
    )

    payload_out: Dict[str, Any] = {
        "reply": response.get("output", ""),
        "actions": response.get("actions", []),
        "tool_runs": response.get("tool_runs", []),
    }
    return JSONResponse(payload_out)


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.pop(SESSION_USER_KEY, None)
    request.session.pop(SESSION_ID_KEY, None)
    request.session.pop(SESSION_CHANNEL, None)
    return RedirectResponse(url="/integration/login", status_code=status.HTTP_302_FOUND)
