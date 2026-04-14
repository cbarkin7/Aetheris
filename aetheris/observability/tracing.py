"""
LangSmith observability integration.
All LangChain/LangGraph calls are automatically traced when LANGCHAIN_TRACING_V2=true.
This module provides helpers for custom run metadata and client access.
"""
import logging
import os
from typing import Any

from aetheris.config import get_settings

logger = logging.getLogger(__name__)


def configure_langsmith() -> None:
    """Set LangSmith environment variables from Settings (idempotent)."""
    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", str(settings.langchain_tracing_v2).lower())
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        logger.info("LangSmith tracing configured for project '%s'", settings.langsmith_project)
    else:
        logger.warning("LANGSMITH_API_KEY not set — tracing disabled")


def get_langsmith_client():
    """Return an authenticated LangSmith Client, or None if not configured."""
    settings = get_settings()
    if not settings.langsmith_api_key:
        return None
    try:
        from langsmith import Client
        return Client(api_key=settings.langsmith_api_key)
    except Exception as exc:
        logger.error("Failed to create LangSmith client: %s", exc)
        return None


def get_langsmith_callbacks() -> list[Any]:
    """
    Return LangSmith callback handlers for injection into graph invocations.
    Returns empty list if LangSmith is not configured (tracing still works via env vars).
    """
    settings = get_settings()
    if not settings.langsmith_api_key or not settings.langchain_tracing_v2:
        return []
    try:
        from langsmith.run_helpers import LangSmithTracer
        return [LangSmithTracer(project_name=settings.langsmith_project)]
    except ImportError:
        # Tracing works via env vars even without explicit callback
        return []


def get_recent_runs(limit: int = 20) -> list[dict]:
    """Fetch recent runs from LangSmith for the observability page."""
    client = get_langsmith_client()
    if client is None:
        return []
    settings = get_settings()
    try:
        runs = list(client.list_runs(project_name=settings.langsmith_project, limit=limit))
        return [
            {
                "id": str(r.id),
                "name": r.name,
                "status": r.status,
                "start_time": str(r.start_time),
                "total_tokens": getattr(r, "total_tokens", None),
                "total_cost": getattr(r, "total_cost", None),
            }
            for r in runs
        ]
    except Exception as exc:
        logger.error("Failed to fetch LangSmith runs: %s", exc)
        return []
