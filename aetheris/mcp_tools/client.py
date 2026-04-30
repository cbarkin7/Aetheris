"""
MCP client — bridges LangChain agents to external MCP servers.
MultiServerMCPClient is started once at FastAPI lifespan and stored in app.state.

Servidores disponibles:
  - tavily    → búsqueda web en tiempo real (@modelcontextprotocol/server-tavily)
  - calendar  → Google Calendar (@cocal/google-calendar-mcp)
  - gmail     → Gmail HTTP + Bearer (servidor externo en GMAIL_MCP_URL)
  - drive     → Google Drive (@modelcontextprotocol/server-gdrive)

IMPORTANTE — ciclo de vida de los clientes:
  get_mcp_tools() devuelve (mcp_clients, mcp_tools).
  Los clientes DEBEN guardarse en app.state para mantener los procesos npx vivos.
  Si el cliente es garbage-collected, la conexión stdio se cierra y tool.ainvoke()
  falla con un error de conexión aunque las tools parezcan cargadas.
"""
import logging
from pathlib import Path
from typing import Any

from aetheris.config import get_settings
from aetheris.mcp_tools.google_tools import (
    calendar_server_config,
    drive_server_config,
    ensure_google_credentials_files,
    gmail_server_config,
)
from aetheris.mcp_tools.tavily_tools import get_tavily_server_config

logger = logging.getLogger(__name__)


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


async def get_mcp_tools(
    include_tavily: bool = True,
    include_google: bool = True,
) -> tuple[list[Any], list[Any]]:
    """
    Inicia los servidores MCP configurados y devuelve (mcp_clients, mcp_tools).

    Cada servidor se conecta de forma independiente: si uno falla, el error
    queda aislado y no cancela los demás.

    Returns:
        (mcp_clients, mcp_tools)
        - mcp_clients: lista de MultiServerMCPClient activos.
          DEBEN guardarse en app.state para mantener los procesos npx vivos.
        - mcp_tools: lista plana de LangChain tools listas para invocar.
    """
    logger.info(
        "[MCP] → get_mcp_tools | inicio | tavily=%s google=%s",
        include_tavily, include_google,
    )

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401
    except ImportError:
        logger.error(
            "[MCP] → get_mcp_tools | langchain-mcp-adapters no instalado "
            "| pip install langchain-mcp-adapters"
        )
        return [], []

    settings = get_settings()
    servers: dict[str, dict] = {}

    # -- Tavily ----------------------------------------------------------------
    if include_tavily and settings.tavily_api_key:
        servers["tavily"] = get_tavily_server_config()
        logger.debug("[MCP] → get_mcp_tools | servidor Tavily configurado")
    else:
        logger.info("[MCP] → get_mcp_tools | Tavily omitido (sin API key o desactivado)")

    # -- Google Calendar + Gmail -----------------------------------------------
    google_available = bool(
        settings.google_refresh_token
        and Path(settings.google_client_secret_file).exists()
    )

    if include_google and google_available:
        creds_ok = ensure_google_credentials_files()
        if creds_ok:
            servers["calendar"] = calendar_server_config()
            servers["gmail"] = gmail_server_config()
            servers["drive"] = drive_server_config()
            logger.debug("[MCP] → get_mcp_tools | servidores Calendar, Gmail y Drive configurados")
        else:
            logger.warning(
                "[MCP] → get_mcp_tools | Google omitido "
                "(no se pudieron crear los ficheros de credenciales)"
            )
    else:
        logger.info(
            "[MCP] → get_mcp_tools | Google omitido "
            "(sin GOOGLE_CLIENT_SECRET_FILE / GOOGLE_REFRESH_TOKEN)"
        )

    if not servers:
        logger.warning(
            "[MCP] → get_mcp_tools | sin servidores configurados → agente sin herramientas externas"
        )
        return [], []

    logger.info(
        "[MCP] → get_mcp_tools | servidores=%s | iniciando conexión independiente por servidor",
        list(servers.keys()),
    )

    # Cada servidor se conecta por separado para que un fallo no cancele los demás.
    # El cliente se guarda en mcp_clients para mantener el proceso npx vivo:
    # si el cliente es GC'd, la conexión stdio se cierra y tool.ainvoke() falla.
    mcp_clients: list[Any] = []
    all_tools: list[Any] = []

    for name, config in servers.items():
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            client = MultiServerMCPClient({name: config})
            tools = await client.get_tools()
            mcp_clients.append(client)   # mantener referencia viva
            all_tools.extend(tools)
            logger.info(
                "[MCP] → get_mcp_tools | servidor='%s' | tools=%d",
                name, len(tools),
            )
        except BaseException as exc:
            cause = _unwrap_exception(exc)
            logger.error(
                "[MCP] → get_mcp_tools | servidor='%s' | fallido | causa: %s",
                name, cause,
            )

    tool_names = [t.name for t in all_tools]
    logger.info(
        "[MCP] → get_mcp_tools | completado | clients=%d tools=%d nombres=%s",
        len(mcp_clients), len(all_tools), tool_names,
    )
    return mcp_clients, all_tools
