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


async def get_mcp_tools(include_tavily: bool = True, include_google: bool = True) -> list[Any]:
    """
    Start MCP servers and return a flat list of LangChain-compatible BaseTool objects.
    Falls back gracefully if a server fails to start (missing API key, npx not installed, etc.).
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.error("langchain-mcp-adapters is not installed. Run: pip install langchain-mcp-adapters")
        return []

    settings = get_settings()
    servers: dict[str, dict] = {}

    if include_tavily and settings.tavily_api_key:
        servers["tavily"] = get_tavily_server_config()
    else:
        logger.info("Tavily MCP skipped (no API key or disabled)")

    if include_google and settings.google_client_id and settings.google_refresh_token:
        servers["google"] = get_google_server_config()
    else:
        logger.info("Google MCP skipped (no credentials or disabled)")

    if not servers:
        logger.warning("No MCP servers configured — agent will run without external tools")
        return []

    tools: list[Any] = []
    client = MultiServerMCPClient(servers)
    try:
        tools = await client.get_tools()
        logger.info("Loaded %d MCP tools from %d server(s)", len(tools), len(servers))
    except Exception as exc:
        logger.error("Failed to load MCP tools: %s", exc)

    return tools


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
        logger.error("langchain-mcp-adapters is not installed.")
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
