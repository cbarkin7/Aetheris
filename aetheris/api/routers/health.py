"""Health check endpoints."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends
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
