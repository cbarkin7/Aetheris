"""Health check endpoints."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from langchain_chroma import Chroma

from aetheris.api.schemas import HealthResponse, LangSmithHealthResponse
from aetheris.config import Settings, get_settings
from aetheris.api.dependencies import get_app_settings

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    # Chroma check
    chroma_ok = False
    try:
        store = Chroma(
            collection_name="aetheris",
            persist_directory=str(settings.chroma_persist_path),
        )
        store._collection.count()
        chroma_ok = True
    except Exception:
        pass

    # SQLite check
    sqlite_ok = False
    try:
        path = settings.sqlite_checkpoints_path
        if Path(path).exists():
            conn = sqlite3.connect(path)
            conn.execute("SELECT 1")
            conn.close()
            sqlite_ok = True
        else:
            sqlite_ok = True  # File not yet created is fine
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="0.1.0",
        chroma_ok=chroma_ok,
        sqlite_ok=sqlite_ok,
        app_env=settings.app_env,
    )


@router.get("/google")
async def health_google(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """
    Estado de las credenciales y herramientas MCP de Google Workspace.

    Útil para diagnosticar problemas de conexión con Calendar y Gmail.
    """
    # -- Credenciales en disco -------------------------------------------------
    cal_token = Path("data/google/.calendar-token.json")
    gmail_token = Path("data/google/.gmail-token.json")

    credentials = {
        "client_secret_file_exists": Path(settings.google_client_secret_file).exists(),
        "refresh_token_set": bool(settings.google_refresh_token),
        "calendar_token_exists": cal_token.exists(),
        "gmail_token_exists": gmail_token.exists(),
    }

    # -- Herramientas MCP cargadas en app.state --------------------------------
    mcp_tools = getattr(request.app.state, "mcp_tools", [])
    mcp_clients = getattr(request.app.state, "mcp_clients", [])

    _google_kw = ("google", "calendar", "gmail", "drive", "event", "email", "list_events",
                  "create_event", "send", "draft", "label", "thread", "message")
    google_tools = [
        t.name for t in mcp_tools
        if any(kw in t.name.lower() for kw in _google_kw)
    ]

    mcp_info = {
        "total_tools_loaded": len(mcp_tools),
        "google_tools": google_tools,
        "google_tools_count": len(google_tools),
        "clients_alive": len(mcp_clients),
    }

    # -- Estado global ---------------------------------------------------------
    creds_ok = credentials["client_secret_file_exists"] and credentials["refresh_token_set"]
    tools_ok = len(google_tools) > 0

    if creds_ok and tools_ok:
        status = "ok"
    elif creds_ok and not tools_ok:
        status = "partial"   # credenciales OK pero MCP no cargó
    else:
        status = "error"     # credenciales faltantes

    return {
        "status": status,
        "credentials": credentials,
        "mcp": mcp_info,
    }


@router.get("/langsmith", response_model=LangSmithHealthResponse)
def langsmith_health(settings: Settings = Depends(get_app_settings)) -> LangSmithHealthResponse:
    try:
        from langsmith import Client
        client = Client(api_key=settings.langsmith_api_key)
        projects = list(client.list_projects())
        return LangSmithHealthResponse(
            langsmith_connected=True,
            project_name=settings.langsmith_project,
        )
    except Exception as exc:
        return LangSmithHealthResponse(
            langsmith_connected=False,
            project_name=settings.langsmith_project,
            error=str(exc),
        )
