"""
AETHERIS — Memoria a corto plazo con mem0.ai

mem0 gestiona la memoria conversacional y de sesión, extrayendo automáticamente
hechos relevantes de las conversaciones y permitiendo búsquedas semánticas.

Modos de operación:
  - Cloud: usa mem0.ai SaaS (requiere MEM0_API_KEY)
  - Local: usa MemoryClient local con Chroma como vector store
"""
import logging
from typing import Any

from aetheris.config import get_settings

logger = logging.getLogger(__name__)

_mem0_client: Any = None


def _build_mem0_cloud():
    """Inicializa el cliente mem0 en modo cloud."""
    from mem0 import MemoryClient

    settings = get_settings()
    kwargs = {"api_key": settings.mem0_api_key}
    if settings.mem0_org_id:
        kwargs["org_id"] = settings.mem0_org_id
    if settings.mem0_project_id:
        kwargs["project_id"] = settings.mem0_project_id

    return MemoryClient(**kwargs)


def _build_mem0_local():
    """Inicializa mem0 en modo local con Chroma."""
    from mem0 import Memory

    settings = get_settings()
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "aetheris_mem0",
                "path": str(settings.chroma_persist_path / "mem0"),
            },
        },
    }
    # Si hay clave de OpenAI, usar embeddings de OpenAI
    if settings.openai_api_key:
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "api_key": settings.openai_api_key,
            },
        }
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": settings.llm_model,
                "api_key": settings.openai_api_key,
            },
        }

    return Memory.from_config(config)


def get_mem0_client() -> Any:
    """Devuelve el cliente mem0 singleton (cloud o local según configuración)."""
    global _mem0_client
    if _mem0_client is None:
        settings = get_settings()
        if settings.mem0_cloud_mode:
            _mem0_client = _build_mem0_cloud()
            logger.info("mem0 inicializado en modo cloud")
        else:
            _mem0_client = _build_mem0_local()
            logger.info("mem0 inicializado en modo local con Chroma")
    return _mem0_client


def add_conversation_memory(
    messages: list[dict[str, str]],
    user_id: str,
    session_id: str | None = None,
) -> dict:
    """
    Registra una conversación en mem0 para extracción automática de hechos.

    Args:
        messages: Lista de dicts con {role: "user"|"assistant", content: "..."}
        user_id: Identificador del usuario.
        session_id: Identificador de sesión (opcional).

    Returns:
        Resultado de mem0.add() con los hechos extraídos.
    """
    client = get_mem0_client()
    kwargs: dict[str, Any] = {"user_id": user_id}
    if session_id:
        kwargs["session_id"] = session_id

    try:
        result = client.add(messages, **kwargs)
        logger.info(
            "mem0: registrada conversación para user='%s' (session='%s')",
            user_id, session_id,
        )
        return result
    except Exception as exc:
        logger.error("mem0: error al registrar conversación: %s", exc)
        return {}


def search_memory(
    query: str,
    user_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Busca memorias relevantes para la consulta del usuario.

    Returns:
        Lista de memorias: [{memory, id, score, ...}]
    """
    client = get_mem0_client()
    try:
        # La API v2 de mem0 cloud (MemoryClient) exige el parámetro `filters` explícito;
        # pasar solo `user_id` produce 400 "Filters are required and cannot be empty".
        # En modo local (Memory) el cliente acepta `user_id` directamente como argumento.
        settings = get_settings()
        if settings.mem0_cloud_mode:
            results = client.search(
                query=query,
                filters={"AND": [{"user_id": user_id}]},
                limit=limit,
            )
        else:
            results = client.search(query=query, user_id=user_id, limit=limit)

        # mem0 cloud devuelve lista directamente; local puede devolver dict
        if isinstance(results, dict):
            return results.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as exc:
        logger.error("mem0: error en búsqueda de memoria: %s", exc)
        return []


def get_all_memories(user_id: str) -> list[dict]:
    """Obtiene todas las memorias almacenadas para un usuario."""
    client = get_mem0_client()
    try:
        result = client.get_all(user_id=user_id)
        if isinstance(result, dict):
            return result.get("results", [])
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.error("mem0: error al obtener memorias: %s", exc)
        return []


def delete_all_memories(user_id: str) -> None:
    """Elimina todas las memorias de un usuario."""
    client = get_mem0_client()
    try:
        memories = get_all_memories(user_id)
        for mem in memories:
            mem_id = mem.get("id")
            if mem_id:
                client.delete(mem_id)
        logger.info("mem0: eliminadas todas las memorias para user='%s'", user_id)
    except Exception as exc:
        logger.error("mem0: error al eliminar memorias: %s", exc)
