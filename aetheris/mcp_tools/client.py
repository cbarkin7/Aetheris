"""
MCP client — bridges LangChain agents to external MCP servers.
MultiServerMCPClient is started once at FastAPI lifespan and stored in app.state.

Servidores disponibles:
  - tavily    → búsqueda web en tiempo real (@modelcontextprotocol/server-tavily)
  - calendar  → Google Calendar (@cocal/google-calendar-mcp)             [stdio, cmd /c npx]
  - drive     → Google Drive (@piotr-agier/google-drive-mcp)              [stdio, cmd /c npx]
  - gmail     → Gmail Python MCP nativo (gmail_mcp_server.py)             [stdio]

IMPORTANTE — ciclo de vida de los clientes:
  get_mcp_tools() devuelve (mcp_clients, mcp_tools).
  Los clientes DEBEN guardarse en app.state para mantener los procesos npx vivos.
  Si el cliente es garbage-collected, la conexión stdio se cierra y tool.ainvoke()
  falla con un error de conexión aunque las tools parezcan cargadas.

Configuración Google:
  Los servidores Calendar y Drive usan get_google_env() de mcp_client.py, que
  construye rutas absolutas ancladas al directorio del módulo. Esto evita
  problemas con el CWD del proceso uvicorn y coincide con los tests funcionales.
  En Windows se invoca npx a través de 'cmd /c npx' porque npx es un script .cmd
  que no se puede lanzar directamente como subprocess.
"""
import logging
import os
from pathlib import Path
from typing import Any

from aetheris.config import get_settings
from aetheris.mcp_tools.google_tools import ensure_google_credentials_files
from aetheris.mcp_tools.tavily_tools import get_tavily_server_config

logger = logging.getLogger(__name__)


def _unwrap_exception(exc: BaseException) -> str:
    """
    Extrae el mensaje de la sub-excepción raíz de un ExceptionGroup.
    Si no es un grupo, devuelve str(exc) directamente.
    """
    if hasattr(exc, "exceptions") and exc.exceptions:
        causes = [_unwrap_exception(e) for e in exc.exceptions]
        return " | ".join(causes)
    return str(exc)


def _google_server_configs(settings) -> dict[str, dict]:
    """
    Construye los configs de los tres servidores MCP de Google usando:
    - get_google_env() de mcp_client.py → rutas absolutas, mismo env que los tests funcionales
    - cmd /c npx en Windows para Calendar y Drive (npx es .cmd en Windows)
    - Servidor Python nativo (stdio) para Gmail — gmail_mcp_server.py

    Cada servidor se devuelve como entrada separada para permitir
    que get_mcp_tools() los conecte de forma independiente (fault isolation).
    """
    from aetheris.mcp_tools.mcp_client import get_google_env
    from aetheris.mcp_tools.google_tools import gmail_server_config

    google_env = get_google_env()

    # Windows: npx es un .cmd — necesita cmd /c para ejecutarse como subprocess
    if os.name == "nt":
        npx_cmd, npx_args_prefix = "cmd", ["/c", "npx", "-y"]
    else:
        npx_cmd, npx_args_prefix = "npx", ["-y"]

    configs: dict[str, dict] = {}

    # Calendar
    configs["calendar"] = {
        "transport": "stdio",
        "command": npx_cmd,
        "args": npx_args_prefix + ["@cocal/google-calendar-mcp"],
        "env": google_env,
    }

    # Drive — @piotr-agier/google-drive-mcp: search, listFolder, uploadFile,
    # deleteItem, moveItem, renameItem, copyFile, downloadFile, Docs/Sheets/Slides
    configs["drive"] = {
        "transport": "stdio",
        "command": npx_cmd,
        "args": npx_args_prefix + ["@piotr-agier/google-drive-mcp"],
        "env": google_env,
    }

    # Gmail — servidor MCP Python nativo (stdio), reemplaza el servidor npm HTTP.
    # gmail_server_config() prepara el token con client_id/secret y devuelve
    # el config para lanzar gmail_mcp_server.py como subproceso.
    try:
        configs["gmail"] = gmail_server_config()
    except Exception as exc:
        logger.warning("[MCP] → _google_server_configs | Gmail omitido: %s", exc)

    return configs


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

    # -- Google Calendar + Drive + Gmail ---------------------------------------
    # Usa get_google_env() (rutas absolutas) y cmd /c npx en Windows,
    # replicando exactamente el config validado por los tests funcionales.
    #
    # IMPORTANTE: settings.google_client_secret_file es una ruta relativa
    # ("data/google/client_secret_aetheris.json"). Path.exists() la resuelve
    # contra el CWD del proceso — si uvicorn no arranca desde la raíz del
    # proyecto, el fichero no se encuentra aunque exista.
    # Solución: resolver la ruta usando get_project_root() de mcp_client.py,
    # que ancla la ruta al directorio del módulo Python (independiente del CWD).
    from aetheris.mcp_tools.mcp_client import get_project_root
    _project_root = get_project_root()
    _secret_path = Path(settings.google_client_secret_file)
    if not _secret_path.is_absolute():
        _secret_path = _project_root / _secret_path

    _secret_exists = _secret_path.exists()
    _rt_set = bool(settings.google_refresh_token)
    logger.info(
        "[MCP] → get_mcp_tools | Google check | secret_file='%s' exists=%s refresh_token_set=%s",
        _secret_path, _secret_exists, _rt_set,
    )
    google_available = _secret_exists and _rt_set

    if include_google and google_available:
        creds_ok = ensure_google_credentials_files()
        if creds_ok:
            google_configs = _google_server_configs(settings)
            servers.update(google_configs)
            logger.debug(
                "[MCP] → get_mcp_tools | servidores Google configurados: %s",
                list(google_configs.keys()),
            )
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

    # ---------------------------------------------------------------------------
    # Blocklist de tools por servidor.
    # Solo se excluyen los nombres listados; el resto se acepta sin filtro.
    #
    # Motivo de exclusión en "drive":
    #   @piotr-agier/google-drive-mcp expone también herramientas de Calendar
    #   (listCalendars, getCalendarEvents, …) que duplican las de
    #   @cocal/google-calendar-mcp con nombres menos específicos. Excluirlas
    #   evita que el LLM use la versión inferior cuando tiene la de Calendar.
    # ---------------------------------------------------------------------------
    _TOOL_BLOCKLIST: dict[str, set[str]] = {
        "tavily":   set(),  # sin exclusiones
        "calendar": set(),  # sin exclusiones
        "gmail":    set(),  # sin exclusiones
        "drive":    set(),  # sin exclusiones — añadir nombres aquí para filtrar
        # Ejemplo de tools a excluir en drive si se quieren evitar duplicados
        # de @cocal/google-calendar-mcp:
        # "drive": {
        #     "listCalendars", "getCalendarEvents",
        #     "createCalendarEvent", "updateCalendarEvent", "deleteCalendarEvent",
        # },
    }

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

            # Aplicar blocklist: excluir solo las tools no deseadas
            blocked = _TOOL_BLOCKLIST.get(name, set())
            if blocked:
                dropped = [t.name for t in tools if t.name in blocked]
                tools = [t for t in tools if t.name not in blocked]
                if dropped:
                    logger.info(
                        "[MCP] → get_mcp_tools | servidor='%s' | tools excluidas (blocklist): %s",
                        name, dropped,
                    )

            mcp_clients.append(client)
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
