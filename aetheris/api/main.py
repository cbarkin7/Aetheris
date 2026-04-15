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
from aetheris.logging_config import setup_logging
from aetheris.observability.tracing import configure_langsmith

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    settings = get_settings()

    # 0. Configurar logging (debe ser lo primero)
    setup_logging(level=settings.log_level)
    logger.info("[SISTEMA] → lifespan | inicio | env=%s version=0.1.0", settings.app_env)

    # 1. Configure LangSmith tracing
    configure_langsmith()
    logger.info("[SISTEMA] → configure_langsmith | completado | tracing=%s", settings.langchain_tracing_v2)

    # 2. Ensure data directories exist
    for path_str in [
        settings.chroma_persist_dir,
        str(Path(settings.sqlite_checkpoints_path).parent),
        str(Path(settings.sqlite_memory_path).parent),
        settings.uploads_dir,
    ]:
        Path(path_str).mkdir(parents=True, exist_ok=True)
    logger.debug("[SISTEMA] → directorios de datos | verificados")

    # 3. Start MCP servers and load tools
    mcp_tools: list = []
    logger.info("[MCP] → get_mcp_tools | inicio")
    try:
        from aetheris.mcp.client import get_mcp_tools
        mcp_tools = await get_mcp_tools()
        logger.info("[MCP] → get_mcp_tools | completado | tools=%d nombres=%s",
                    len(mcp_tools), [t.name for t in mcp_tools])
    except Exception as exc:
        logger.warning("[MCP] → get_mcp_tools | fallido (continuando sin herramientas) | error=%s", exc)

    app.state.mcp_tools = mcp_tools

    # 4. Build and compile the LangGraph agent
    logger.info("[SISTEMA] → build_graph | inicio | mcp_tools=%d", len(mcp_tools))
    from aetheris.agent.graph import build_graph
    from aetheris.memory.checkpointer import create_async_checkpointer
    checkpointer = await create_async_checkpointer()
    app.state.graph = build_graph(mcp_tools=mcp_tools, checkpointer=checkpointer)
    logger.info("[SISTEMA] → build_graph | completado | grafo listo para recibir peticiones")

    yield

    # Shutdown
    logger.info("[SISTEMA] → lifespan | apagado | AETHERIS detenido")


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
