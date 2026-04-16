"""
MCP client — bridges LangChain agents to external MCP servers.
MultiServerMCPClient is started once at FastAPI lifespan and stored in app.state.
"""
import logging
from typing import Any

from aetheris.config import get_settings
from aetheris.mcp.google_tools import get_google_server_config
from aetheris.mcp.tavily_tools import get_tavily_server_config

logger = logging.getLogger(__name__)


async def _load_server(name: str, config: dict) -> list[Any]:
    """
    Conecta a un único servidor MCP y devuelve sus herramientas.

    Cada servidor se conecta de forma independiente: si uno falla, el error
    queda aislado y no cancela los demás (evita el ExceptionGroup de TaskGroup
    que se propaga cuando MultiServerMCPClient agrupa todos los servidores juntos).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    try:
        client = MultiServerMCPClient({name: config})
        tools = await client.get_tools()
        logger.info("[MCP] → _load_server | servidor='%s' | tools=%d", name, len(tools))
        return tools
    except BaseException as exc:
        # BaseException captura también ExceptionGroup (Python 3.11+)
        cause = _unwrap_exception(exc)
        logger.error(
            "[MCP] → _load_server | servidor='%s' | fallido | causa: %s",
            name, cause,
        )
        return []


def _unwrap_exception(exc: BaseException) -> str:
    """
    Extrae el mensaje de la sub-excepción raíz de un ExceptionGroup.
    Si no es un grupo, devuelve str(exc) directamente.
    """
    # ExceptionGroup / BaseExceptionGroup (Python 3.11+)
    if hasattr(exc, "exceptions") and exc.exceptions:
        causes = [_unwrap_exception(e) for e in exc.exceptions]
        return " | ".join(causes)
    return str(exc)


async def get_mcp_tools(include_tavily: bool = True, include_google: bool = True) -> list[Any]:
    """
    Inicia los servidores MCP configurados y devuelve la lista plana de herramientas.

    Cada servidor se conecta de forma independiente para que un fallo aislado
    (npx no instalado, credenciales inválidas, timeout) no impida cargar los
    demás servidores disponibles.
    """
    logger.info(
        "[MCP] → get_mcp_tools | inicio | tavily=%s google=%s",
        include_tavily, include_google,
    )

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401
    except ImportError:
        logger.error("[MCP] → get_mcp_tools | langchain-mcp-adapters no instalado | pip install langchain-mcp-adapters")
        return []

    settings = get_settings()
    servers: dict[str, dict] = {}

    if include_tavily and settings.tavily_api_key:
        servers["tavily"] = get_tavily_server_config()
        logger.debug("[MCP] → get_mcp_tools | servidor Tavily configurado")
    else:
        logger.info("[MCP] → get_mcp_tools | Tavily omitido (sin API key o desactivado)")

    if include_google and settings.google_client_id and settings.google_refresh_token:
        servers["google"] = get_google_server_config()
        logger.debug("[MCP] → get_mcp_tools | servidor Google configurado")
    else:
        logger.info("[MCP] → get_mcp_tools | Google omitido (sin credenciales o desactivado)")

    if not servers:
        logger.warning("[MCP] → get_mcp_tools | sin servidores configurados → agente sin herramientas externas")
        return []

    logger.info("[MCP] → get_mcp_tools | servidores=%s | iniciando conexión independiente por servidor", list(servers.keys()))

    # Cada servidor se conecta por separado: un fallo no cancela los demás.
    all_tools: list[Any] = []
    for name, config in servers.items():
        tools = await _load_server(name, config)
        all_tools.extend(tools)

    tool_names = [t.name for t in all_tools]
    logger.info(
        "[MCP] → get_mcp_tools | completado | tools=%d nombres=%s",
        len(all_tools), tool_names,
    )
    return all_tools


async def get_mcp_tools_persistent(
    include_tavily: bool = True,
    include_google: bool = True,
) -> tuple[Any, list[Any]]:
    """
    Return (client_context_manager, tools) for long-lived FastAPI lifespan use.
    The caller is responsible for entering and exiting the context manager.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.error("[MCP] → get_mcp_tools_persistent | langchain-mcp-adapters no instalado.")
        return None, []

    settings = get_settings()
    servers: dict[str, dict] = {}

    if include_tavily and settings.tavily_api_key:
        servers["tavily"] = get_tavily_server_config()
    if include_google and settings.google_client_id and settings.google_refresh_token:
        servers["google"] = get_google_server_config()

    if not servers:
        return None, []

    client = MultiServerMCPClient(servers)
    return client, servers
