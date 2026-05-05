"""
AETHERIS — Memoria a largo plazo con SQLite + Vector DB (Chroma).

Dos capas:
  - SQLite (user_memory): almacén clave-valor para preferencias explícitas del usuario.
  - Chroma (long_term_facts): almacén vectorial para hechos extraídos de conversaciones,
    permitiendo búsqueda semántica y trazabilidad temporal.
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aetheris.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capa 1: SQLite clave-valor (preferencias de usuario)
# ---------------------------------------------------------------------------

_DDL_USER_MEMORY = """
CREATE TABLE IF NOT EXISTS user_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, key)
)
"""

_DDL_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    settings = get_settings()
    path = db_path or settings.sqlite_memory_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_DDL_USER_MEMORY)
    conn.execute(_DDL_CONVERSATIONS)
    conn.commit()
    return conn


def load_user_memory(user_id: str, db_path: str | None = None) -> dict:
    """Devuelve dict {clave: valor} para el usuario dado."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def upsert_user_memory(user_id: str, updates: dict, db_path: str | None = None) -> None:
    """Inserta o reemplaza entradas de memoria para el usuario dado."""
    if not updates:
        return
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO user_memory (user_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [(user_id, k, str(v)) for k, v in updates.items()],
        )
        conn.commit()
        logger.info("Actualizadas %d entradas de memoria para user='%s'", len(updates), user_id)
    finally:
        conn.close()


def delete_user_memory(user_id: str, db_path: str | None = None) -> None:
    """Elimina toda la memoria de un usuario."""
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Historial de conversaciones (tabla conversations)
# ---------------------------------------------------------------------------

def save_conversation(
    thread_id: str,
    user_id: str,
    title: str,
    db_path: str | None = None,
) -> None:
    """
    Registra o actualiza una conversación en el historial.
    Si el thread ya existe, actualiza el updated_at (no el título).
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO conversations (thread_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, user_id, title[:120]),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("save_conversation: no crítico | %s", exc)
    finally:
        conn.close()


def list_conversations(
    user_id: str,
    limit: int = 30,
    db_path: str | None = None,
) -> list[dict]:
    """
    Devuelve las últimas `limit` conversaciones del usuario, ordenadas por updated_at DESC.
    Cada elemento: {thread_id, title, created_at, updated_at}
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "thread_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]
    finally:
        conn.close()


def delete_conversation(thread_id: str, db_path: str | None = None) -> dict:
    """
    Elimina una conversación del historial Y sus checkpoints LangGraph.

    Borra de dos bases de datos:
    - memory.db     → tabla `conversations`
    - checkpoints.db → tablas `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`

    Returns:
        Dict con `{"memory_deleted": bool, "checkpoints_deleted": bool}`.
    """
    settings = get_settings()

    # 1. Eliminar de memory.db
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        conn.commit()
        memory_deleted = True
    except Exception as exc:
        logger.warning("delete_conversation: error en memory.db | %s", exc)
        memory_deleted = False
    finally:
        conn.close()

    # 2. Eliminar checkpoints de checkpoints.db.
    # Se usa sqlite3 estándar (no aiosqlite) para poder llamar desde código síncrono.
    # SQLite permite múltiples conexiones simultáneas al mismo fichero; el acceso
    # de escritura puntual no interfiere con la conexión de lectura/escritura continua
    # del AsyncSqliteSaver de LangGraph.
    checkpoints_deleted = False
    cp_path = settings.sqlite_checkpoints_path
    if Path(cp_path).exists():
        try:
            cp_conn = sqlite3.connect(cp_path, check_same_thread=False, timeout=10)
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                try:
                    cp_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                except sqlite3.OperationalError:
                    pass  # tabla no existe aún en esta versión de LangGraph — ignorar
            cp_conn.commit()
            cp_conn.close()
            checkpoints_deleted = True
            logger.info(
                "delete_conversation | checkpoints eliminados | thread='%s'", thread_id
            )
        except Exception as exc:
            logger.warning(
                "delete_conversation: error limpiando checkpoints | thread='%s' | %s",
                thread_id, exc,
            )

    logger.info(
        "delete_conversation | completado | thread='%s' memory=%s checkpoints=%s",
        thread_id, memory_deleted, checkpoints_deleted,
    )
    return {"memory_deleted": memory_deleted, "checkpoints_deleted": checkpoints_deleted}


# ---------------------------------------------------------------------------
# Capa 2: Chroma vectorial (hechos a largo plazo con búsqueda semántica)
# ---------------------------------------------------------------------------

_LONG_TERM_COLLECTION = "aetheris_long_term_facts"
_long_term_store_cache: Any = None  # singleton — se invalida si se pasan embeddings custom


def _get_long_term_store(embeddings: Any = None):
    """
    Devuelve la instancia Chroma para hechos a largo plazo.
    Se cachea para evitar recrear embeddings + colección en cada llamada.
    Los embeddings custom (tests) siempre crean una instancia nueva.
    """
    global _long_term_store_cache
    if embeddings is not None:
        # Embeddings personalizados (ej. tests) → instancia nueva sin cachear
        from langchain_chroma import Chroma
        settings = get_settings()
        return Chroma(
            collection_name=_LONG_TERM_COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_path),
        )

    if _long_term_store_cache is None:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        settings = get_settings()
        emb = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        _long_term_store_cache = Chroma(
            collection_name=_LONG_TERM_COLLECTION,
            embedding_function=emb,
            persist_directory=str(settings.chroma_persist_path),
        )
        logger.debug("Chroma long-term store inicializado (colección='%s')", _LONG_TERM_COLLECTION)

    return _long_term_store_cache


def store_long_term_fact(
    user_id: str,
    fact: str,
    source: str = "conversation",
    embeddings: Any = None,
) -> str:
    """
    Almacena un hecho en la base vectorial a largo plazo.

    Returns:
        ID del documento almacenado.
    """
    from langchain_core.documents import Document

    store = _get_long_term_store(embeddings)
    doc = Document(
        page_content=fact,
        metadata={
            "user_id": user_id,
            "source": source,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    ids = store.add_documents([doc])
    doc_id = ids[0] if ids else ""
    logger.info("Hecho almacenado para user='%s': '%s…' (id=%s)", user_id, fact[:50], doc_id[:8])
    return doc_id


def search_long_term_facts(
    query: str,
    user_id: str,
    k: int = 5,
    embeddings: Any = None,
) -> list[dict]:
    """
    Busca hechos relevantes en la memoria a largo plazo.

    Returns:
        Lista de {content, source, score, stored_at}.
    """
    store = _get_long_term_store(embeddings)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    filtered = []
    for doc, score in results:
        if doc.metadata.get("user_id") == user_id:
            filtered.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "score": score,
                "stored_at": doc.metadata.get("stored_at", ""),
            })
    return filtered


# ---------------------------------------------------------------------------
# Extracción de memoria (LLM)
# ---------------------------------------------------------------------------

def extract_memory_updates(messages: list, llm) -> dict:
    """
    Usa el LLM para extraer hechos memorables de mensajes recientes.
    Devuelve un dict {clave: valor} para persistir, o {} si no hay nada notable.
    """
    from langchain_core.messages import HumanMessage
    from aetheris.agent.prompts import MEMORY_EXTRACTION_PROMPT

    recent = messages[-6:] if len(messages) > 6 else messages
    conversation_text = "\n".join(
        f"{m.__class__.__name__.replace('Message', '')}: {m.content}"
        for m in recent
        if hasattr(m, "content") and isinstance(m.content, str)
    )

    prompt = MEMORY_EXTRACTION_PROMPT.format(messages=conversation_text)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        logger.debug("Extracción de memoria fallida (no crítico): %s", exc)
        return {}
