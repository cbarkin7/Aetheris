"""
AETHERIS — Factoría LLM con fallback automático OpenAI → Bedrock.

Prioridad:
  1. OpenAI (gpt-4o-mini por defecto)  — proveedor principal
  2. AWS Bedrock / Anthropic Claude    — fallback automático si OpenAI falla

El fallback se activa mediante LangChain `with_fallbacks()`, de forma transparente
para el resto del sistema. El proveedor activo se devuelve para trazabilidad en LangSmith.
"""
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from aetheris.config import get_settings

logger = logging.getLogger(__name__)

# Caché de instancias por configuración (evita reconstruir en cada llamada)
_llm_cache: dict[str, BaseChatModel] = {}


def _cache_key(settings) -> str:
    return f"{settings.openai_api_key[:8]}|{settings.bedrock_model_id}|{settings.bedrock_available}"


def _build_openai_llm(settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )


def _build_bedrock_llm(settings) -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse
    kwargs: dict[str, Any] = {
        "model_id": settings.bedrock_model_id,
        "region_name": settings.aws_region,
        "temperature": settings.llm_temperature,
    }
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return ChatBedrockConverse(**kwargs)


def get_llm(tools: list | None = None) -> tuple[BaseChatModel, str]:
    """
    Devuelve (llm, provider_name) con fallback automático OpenAI → Bedrock.

    Si se pasan herramientas MCP, se vinculan al LLM mediante bind_tools().
    Las instancias base se cachean; solo bind_tools() se aplica por llamada.
    """
    settings = get_settings()
    key = _cache_key(settings)

    if key not in _llm_cache:
        if settings.openai_api_key:
            llm = _build_openai_llm(settings)
            if settings.bedrock_available:
                llm = llm.with_fallbacks([_build_bedrock_llm(settings)])
                logger.info("LLM: OpenAI con fallback a Bedrock (%s)", settings.bedrock_model_id)
            else:
                logger.info("LLM: OpenAI (%s)", settings.llm_model)
            _llm_cache[key] = llm
            provider = "openai"
        elif settings.bedrock_available:
            _llm_cache[key] = _build_bedrock_llm(settings)
            logger.info("LLM: AWS Bedrock (%s)", settings.bedrock_model_id)
            provider = "bedrock"
        else:
            raise RuntimeError(
                "No hay proveedor LLM configurado. "
                "Establece OPENAI_API_KEY o credenciales AWS Bedrock."
            )
        _llm_cache[key + ":provider"] = provider  # type: ignore[assignment]

    llm = _llm_cache[key]
    provider = _llm_cache.get(key + ":provider", "openai")  # type: ignore[assignment]

    if tools:
        llm = llm.bind_tools(tools)

    return llm, provider  # type: ignore[return-value]


def clear_llm_cache() -> None:
    """Limpia la caché de LLM (útil en tests y tras cambiar configuración)."""
    _llm_cache.clear()
