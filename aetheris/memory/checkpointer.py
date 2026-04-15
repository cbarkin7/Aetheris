"""
LangGraph session checkpointer using SQLite (async).
Provides short-term (per-thread) memory for the agent graph.
"""
import logging
from pathlib import Path

from aetheris.config import get_settings

logger = logging.getLogger(__name__)


async def create_async_checkpointer(db_path: str | None = None):
    """
    Create and return an AsyncSqliteSaver connected to the given path.

    Must be called from an async context (e.g. FastAPI lifespan).
    The caller is responsible for keeping the returned object alive for the
    duration of the application — aiosqlite keeps the connection open as long
    as the object is referenced.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    settings = get_settings()
    path = db_path or settings.sqlite_checkpoints_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("[CHECKPOINTER] → create_async_checkpointer | inicio | path='%s'", path)
    conn = await aiosqlite.connect(path)
    checkpointer = AsyncSqliteSaver(conn)
    logger.info("[CHECKPOINTER] → create_async_checkpointer | completado | AsyncSqliteSaver listo")
    return checkpointer
