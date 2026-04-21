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

_DDL = """
CREATE TABLE IF NOT EXISTS user_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, key)
);
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    settings = get_settings()
    path = db_path or settings.sqlite_memory_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_DDL)
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
