"""
AETHERIS FastAPI application.
Lifespan manages: MCP server startup, graph compilation, DB initialization, LangSmith config.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from aetheris.api.middleware import register_middleware
from aetheris.api.routers import chat, documents, health, memory, speech
from aetheris.config import get_settings
from aetheris.observability.tracing import configure_langsmith

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    settings = get_settings()

    # 1. Configure LangSmith tracing
    configure_langsmith()

    # 2. Ensure data directories exist
    for path_str in [
        settings.chroma_persist_dir,
        str(Path(settings.sqlite_checkpoints_path).parent),
        str(Path(settings.sqlite_memory_path).parent),
        settings.uploads_dir,
    ]:
        Path(path_str).mkdir(parents=True, exist_ok=True)

    # 3. Start MCP servers and load tools
    mcp_tools: list = []
    mcp_client = None
    try:
        from aetheris.mcp.client import get_mcp_tools
        mcp_tools = await get_mcp_tools()
        logger.info("MCP tools loaded: %s", [t.name for t in mcp_tools])
    except Exception as exc:
        logger.warning("MCP tools failed to load (continuing without): %s", exc)

    app.state.mcp_tools = mcp_tools

    # 4. Build and compile the LangGraph agent
    from aetheris.agent.graph import build_graph
    from aetheris.memory.checkpointer import create_async_checkpointer
    checkpointer = await create_async_checkpointer()
    app.state.graph = build_graph(mcp_tools=mcp_tools, checkpointer=checkpointer)
    logger.info("Agent graph compiled and ready")

    yield

    # Shutdown
    logger.info("AETHERIS shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AETHERIS",
        description="Autonomous Cognitive Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    register_middleware(app)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(memory.router)
    app.include_router(speech.router)

    return app


app = create_app()
