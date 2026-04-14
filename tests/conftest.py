"""
Shared pytest fixtures for all test levels.
"""
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

# ---------------------------------------------------------------------------
# Environment setup (must happen before any Settings import)
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("LANGSMITH_API_KEY", "ls-test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


# ---------------------------------------------------------------------------
# Mock LLM (no API calls)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm():
    """FakeChatModel that returns a scripted response."""
    return GenericFakeChatModel(messages=iter(["This is a test response."]))


@pytest.fixture
def mock_llm_json():
    """FakeChatModel that returns a JSON string (for memory extraction)."""
    return GenericFakeChatModel(messages=iter(['{"language": "Spanish", "timezone": "UTC+1"}']))


# ---------------------------------------------------------------------------
# Temporary databases
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_sqlite(tmp_path) -> str:
    """Return a path to a temporary SQLite DB file."""
    return str(tmp_path / "test.db")


@pytest.fixture
def temp_chroma_dir(tmp_path) -> str:
    """Return a path to a temporary Chroma persist directory."""
    d = tmp_path / "chroma"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Settings override
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def override_settings(tmp_path, monkeypatch):
    """Redirect all data paths to temp directories during tests."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("SQLITE_CHECKPOINTS_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("SQLITE_MEMORY_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    # Invalidate lru_cache so new env vars take effect
    from aetheris.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Sample documents (fixture files)
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_txt_file(tmp_path) -> Path:
    """A simple text file with known content."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "AETHERIS is an autonomous cognitive agent.\n"
        "It supports RAG, web search, and Google Workspace integration.\n"
        "The system uses LangGraph for agent orchestration.\n"
        "Memory is persisted using SQLite.\n"
        "The RAG pipeline uses Chroma as the vector store.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_md_file(tmp_path) -> Path:
    """A Markdown file with structured content."""
    f = tmp_path / "readme.md"
    f.write_text(
        "# AETHERIS Documentation\n\n"
        "## Architecture\n"
        "AETHERIS uses a three-layer architecture: RAG, MCP, and Observability.\n\n"
        "## Installation\n"
        "Run `pip install -r requirements.txt` to install dependencies.\n\n"
        "## Configuration\n"
        "Copy `.env.example` to `.env` and fill in your API keys.\n",
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client(override_settings):
    """Return a synchronous FastAPI TestClient with mocked graph and checkpointer."""
    from unittest.mock import AsyncMock, MagicMock
    from fastapi.testclient import TestClient

    mock_graph = MagicMock()
    mock_graph.astream_events = AsyncMock(return_value=aiter([]))
    mock_graph.aget_state = AsyncMock(return_value=MagicMock(values={"messages": []}))
    mock_graph.aupdate_state = AsyncMock()

    mock_checkpointer = MagicMock()

    from aetheris.api.main import create_app
    test_app = create_app()

    # Patch async checkpointer and graph build so tests don't need real DB/LLM
    with patch("aetheris.memory.checkpointer.create_async_checkpointer", new=AsyncMock(return_value=mock_checkpointer)), \
         patch("aetheris.agent.graph.build_graph", return_value=mock_graph):
        with TestClient(test_app) as client:
            test_app.state.graph = mock_graph
            test_app.state.mcp_tools = []
            yield client


# ---------------------------------------------------------------------------
# Agent state factory
# ---------------------------------------------------------------------------
@pytest.fixture
def base_agent_state():
    return {
        "messages": [HumanMessage(content="Hello AETHERIS")],
        "thread_id": "test-thread",
        "user_id": "test-user",
        "intent": "plain_llm",
        "rag_context": [],
        "tool_calls_pending": [],
        "hitl_approved": None,
        "user_memory": {},
        "guardrail_passed": None,
        "guardrail_violations": [],
        "llm_provider": "",
        "execution_plan": [],
        "error": None,
    }


def aiter(items):
    """Create an async iterator from a list."""
    async def _gen():
        for item in items:
            yield item
    return _gen()
