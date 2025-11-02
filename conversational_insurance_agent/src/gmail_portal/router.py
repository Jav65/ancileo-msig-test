from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from starlette.concurrency import run_in_threadpool

from ..config import Settings, get_settings
from ..core.orchestrator import ConversationalOrchestrator
from ..services.gmail_ticket_scraper import GmailDataError, fetch_travel_client
from ..state.client_context import ClientDatum
from ..utils.logging import logger


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


router = APIRouter(prefix="/gmail", tags=["gmail-integration"])


SESSION_PROFILE_KEY = "gmail_profile"
SESSION_CREDENTIALS_KEY = "gmail_credentials"
SESSION_STATE_KEY = "gmail_oauth_state"
SESSION_CLIENT_KEY = "gmail_client_payload"
SESSION_ID_KEY = "gmail_session_id"
SESSION_CHANNEL_KEY = "gmail_channel"
DEFAULT_CHANNEL = "gmail_portal"

GOOGLE_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
)


def mount_static(app) -> None:  # pragma: no cover - runtime wiring
    from fastapi.staticfiles import StaticFiles

    app.mount("/gmail/static", StaticFiles(directory=str(STATIC_DIR)), name="gmail-static")


def get_portal_orchestrator(request: Request) -> Optional[ConversationalOrchestrator]:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if isinstance(orchestrator, ConversationalOrchestrator):
        return orchestrator
    return None


def _build_flow(settings: Settings) -> Flow:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )
    redirect_uri = settings.google_redirect_uri
    if not redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing GOOGLE_REDIRECT_URI environment variable.",
        )

    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }

    flow = Flow.from_client_config(client_config, scopes=list(GOOGLE_SCOPES))
    flow.redirect_uri = redirect_uri
    return flow


def _load_credentials(request: Request, settings: Settings) -> Credentials:
    stored = request.session.get(SESSION_CREDENTIALS_KEY)
    if not stored:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticate with Google first.")

    if isinstance(stored, str):
        try:
            stored_data = json.loads(stored)
        except json.JSONDecodeError as exc:
            logger.warning("gmail_portal.credentials_invalid", error=str(exc))
            request.session.pop(SESSION_CREDENTIALS_KEY, None)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials state") from exc
    else:
        stored_data = stored

    credentials = Credentials.from_authorized_user_info(stored_data, scopes=list(GOOGLE_SCOPES))
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(GoogleAuthRequest())
            request.session[SESSION_CREDENTIALS_KEY] = json.loads(credentials.to_json())
        except Exception as exc:  # pragma: no cover - network interaction
            logger.exception("gmail_portal.credentials_refresh_failed", error=str(exc))
            request.session.pop(SESSION_CREDENTIALS_KEY, None)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google session expired") from exc
    return credentials


def _ensure_session_id(request: Request, profile: Dict[str, Any]) -> str:
    session_id = request.session.get(SESSION_ID_KEY)
    if session_id:
        return session_id
    base = profile.get("email") or profile.get("sub") or "gmail-user"
    sanitized = base.replace("@", "_").replace("|", "_") if isinstance(base, str) else str(base)
    session_id = f"gmail-{sanitized}"
    request.session[SESSION_ID_KEY] = session_id
    return session_id


async def _fetch_userinfo(credentials: Credentials) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {credentials.token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get("https://openidconnect.googleapis.com/v1/userinfo", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()


def _serialize_client(client: ClientDatum) -> Dict[str, Any]:
    serializer = getattr(client, "model_dump", None)
    if callable(serializer):
        return serializer(by_alias=True, exclude_none=True)
    return client.dict(by_alias=True, exclude_none=True)  # type: ignore[return-value]


def _deserialize_client(payload: Dict[str, Any]) -> ClientDatum:
    return ClientDatum.model_validate(payload)


async def _load_client(request: Request, credentials: Credentials) -> ClientDatum:
    cached_payload: Optional[Dict[str, Any]] = request.session.get(SESSION_CLIENT_KEY)
    profile: Dict[str, Any] = request.session.get(SESSION_PROFILE_KEY, {})

    if cached_payload:
        try:
            client = _deserialize_client(cached_payload)
            return client
        except Exception as exc:
            logger.warning("gmail_portal.client_deserialize_failed", error=str(exc))
            request.session.pop(SESSION_CLIENT_KEY, None)

    def build_client() -> ClientDatum:
        return fetch_travel_client(credentials, profile=profile)

    try:
        client = await run_in_threadpool(build_client)
    except GmailDataError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    request.session[SESSION_CLIENT_KEY] = _serialize_client(client)
    return client


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    settings = get_settings()
    try:
        _ = _build_flow(settings)
    except HTTPException as exc:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": exc.detail},
            status_code=exc.status_code,
        )

    if SESSION_CREDENTIALS_KEY in request.session:
        return RedirectResponse(url="/gmail/chat", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.get("/authorize")
async def initiate_oauth(request: Request, settings: Settings = Depends(get_settings)) -> RedirectResponse:
    flow = _build_flow(settings)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    request.session[SESSION_STATE_KEY] = state
    return RedirectResponse(url=authorization_url)


@router.get("/callback")
async def oauth_callback(request: Request, settings: Settings = Depends(get_settings)) -> RedirectResponse:
    stored_state = request.session.get(SESSION_STATE_KEY)
    incoming_state = request.query_params.get("state")
    if not stored_state or stored_state != incoming_state:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")

    flow = _build_flow(settings)
    authorization_response = str(request.url)
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as exc:  # pragma: no cover - network interaction
        logger.exception("gmail_portal.oauth_token_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth token exchange failed") from exc

    credentials = flow.credentials
    request.session[SESSION_CREDENTIALS_KEY] = json.loads(credentials.to_json())

    profile = await _fetch_userinfo(credentials)
    request.session[SESSION_PROFILE_KEY] = profile
    request.session[SESSION_CHANNEL_KEY] = DEFAULT_CHANNEL
    request.session.pop(SESSION_CLIENT_KEY, None)
    _ensure_session_id(request, profile)

    logger.info("gmail_portal.login", email=profile.get("email"), sub=profile.get("sub"))
    return RedirectResponse(url="/gmail/chat", status_code=status.HTTP_302_FOUND)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    settings: Settings = Depends(get_settings),
    orchestrator: Optional[ConversationalOrchestrator] = Depends(get_portal_orchestrator),
) -> HTMLResponse:
    credentials = _load_credentials(request, settings)
    profile = request.session.get(SESSION_PROFILE_KEY)
    if not profile:
        profile = await _fetch_userinfo(credentials)
        request.session[SESSION_PROFILE_KEY] = profile

    client = await _load_client(request, credentials)

    session_id = _ensure_session_id(request, profile)
    channel = request.session.get(SESSION_CHANNEL_KEY, DEFAULT_CHANNEL)

    if orchestrator is None:
        from ..core.setup import build_orchestrator

        orchestrator = build_orchestrator()
        request.app.state.orchestrator = orchestrator

    assert orchestrator is not None  # for type checkers
    orchestrator.merge_clients(session_id=session_id, clients=[client], source=channel)

    context = {
        "request": request,
        "profile": profile,
        "client": client,
        "trips": client.trips,
        "channel": channel,
    }
    return templates.TemplateResponse("chat.html", context)


@router.post("/chat/send")
async def chat_send(
    payload: Dict[str, Any],
    request: Request,
    settings: Settings = Depends(get_settings),
    orchestrator: Optional[ConversationalOrchestrator] = Depends(get_portal_orchestrator),
) -> JSONResponse:
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    credentials = _load_credentials(request, settings)
    profile = request.session.get(SESSION_PROFILE_KEY)
    if not profile:
        profile = await _fetch_userinfo(credentials)
        request.session[SESSION_PROFILE_KEY] = profile

    client = await _load_client(request, credentials)
    session_id = _ensure_session_id(request, profile)
    channel = request.session.get(SESSION_CHANNEL_KEY, DEFAULT_CHANNEL)

    if orchestrator is None:
        from ..core.setup import build_orchestrator

        orchestrator = build_orchestrator()
        request.app.state.orchestrator = orchestrator

    assert orchestrator is not None  # for type checkers
    orchestrator.merge_clients(session_id=session_id, clients=[client], source=channel)

    response = await orchestrator.handle_message(
        session_id=session_id,
        user_message=message,
        channel=channel,
    )

    payload_out = {
        "reply": response.get("output", ""),
        "actions": response.get("actions", []),
        "tool_runs": response.get("tool_runs", []),
    }
    return JSONResponse(payload_out)


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.pop(SESSION_CREDENTIALS_KEY, None)
    request.session.pop(SESSION_PROFILE_KEY, None)
    request.session.pop(SESSION_CLIENT_KEY, None)
    request.session.pop(SESSION_ID_KEY, None)
    request.session.pop(SESSION_CHANNEL_KEY, None)
    request.session.pop(SESSION_STATE_KEY, None)
    return RedirectResponse(url="/gmail/login", status_code=status.HTTP_302_FOUND)
