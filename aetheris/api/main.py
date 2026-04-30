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
    # get_mcp_tools() devuelve (clients, tools). Los clientes DEBEN guardarse en
    # app.state: si son GC'd, la conexión stdio al proceso npx se cierra y
    # tool.ainvoke() falla aunque las tools parezcan cargadas.
    mcp_clients: list = []
    mcp_tools: list = []
    logger.info("[MCP] → get_mcp_tools | inicio")
    try:
        from aetheris.mcp_tools.client import get_mcp_tools
        mcp_clients, mcp_tools = await get_mcp_tools()
        logger.info("[MCP] → get_mcp_tools | completado | clients=%d tools=%d nombres=%s",
                    len(mcp_clients), len(mcp_tools), [t.name for t in mcp_tools])
    except Exception as exc:
        logger.warning("[MCP] → get_mcp_tools | fallido (continuando sin herramientas) | error=%s", exc)

    app.state.mcp_clients = mcp_clients   # mantiene los procesos npx vivos
    app.state.mcp_tools = mcp_tools

    # 4. Build and compile the LangGraph agent
    logger.info("[SISTEMA] → build_graph | inicio | mcp_tools=%d", len(mcp_tools))
    from aetheris.agent.graph import build_graph
    from aetheris.memory.checkpointer import create_async_checkpointer
    checkpointer = await create_async_checkpointer()
    app.state.graph = build_graph(mcp_tools=mcp_tools, checkpointer=checkpointer)
    logger.info("[SISTEMA] → build_graph | completado | grafo listo para recibir peticiones")

    # -------------------------------------------------------------------------
    # FIXME: Remove — preview del contenido de la vector DB (debug, quitar antes de release)
    # -------------------------------------------------------------------------
    try:
        from aetheris.rag.retriever import get_vectorstore
        _vs = get_vectorstore()
        _col = _vs._collection
        _data = _col.get(include=["metadatas", "documents"])
        _ids: list = _data.get("ids", [])
        _metas: list = _data.get("metadatas", []) or []
        _docs: list = _data.get("documents", []) or []

        # Agrupar por document_id
        _groups: dict = {}
        for _i, _meta in enumerate(_metas):
            _did = _meta.get("document_id", "unknown")
            if _did not in _groups:
                _groups[_did] = {
                    "filename": _meta.get("filename", "?"),
                    "ingested_at": _meta.get("ingested_at", "sin fecha"),
                    "chunks": [],
                }
            _groups[_did]["chunks"].append({
                "index": _meta.get("chunk_index", _i),
                "content": _docs[_i] if _i < len(_docs) else "",
            })

        logger.info(
            "[CHROMA][PREVIEW] Base de conocimiento: %d fragmentos en %d documento(s)",
            len(_ids), len(_groups),
        )
        for _did, _info in _groups.items():
            _sorted_chunks = sorted(_info["chunks"], key=lambda x: x["index"])
            _preview = ""
            if _sorted_chunks:
                _raw = _sorted_chunks[0]["content"][:200].replace("\n", " ").strip()
                _preview = (_raw + "…") if len(_sorted_chunks[0]["content"]) > 200 else _raw
            logger.info(
                "[CHROMA][PREVIEW]   📄 %-40s | id=%-10s | %3d fragmentos | %s\n"
                "                        ↳ %s",
                _info["filename"], _did[:8] + "…", len(_info["chunks"]),
                _info["ingested_at"][:19] if _info["ingested_at"] != "sin fecha" else "sin fecha",
                _preview,
            )
    except Exception as _exc:
        logger.debug("[CHROMA][PREVIEW] No se pudo leer el estado de Chroma: %s", _exc)
    # FIXME: Remove — fin preview vector DB
    # -------------------------------------------------------------------------

    yield

    # Shutdown — liberar clientes MCP (cierra conexiones stdio a los procesos npx)
    for _client in getattr(app.state, "mcp_clients", []):
        try:
            if hasattr(_client, "aclose"):
                await _client.aclose()
        except Exception:
            pass
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
