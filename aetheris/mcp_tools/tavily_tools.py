"""Tavily MCP server configuration."""
import os

from aetheris.config import get_settings


def get_tavily_server_config() -> dict:
    """Return the Tavily MCP stdio server config for MultiServerMCPClient."""
    settings = get_settings()
    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "tavily-mcp"],
        # Se hereda todo el entorno del proceso padre (PATH incluido) para que
        # npx pueda localizar el ejecutable de Node.js. Sin esto, el subproceso
        # no encuentra node y el servidor MCP falla con "Connection closed".
        "env": {**os.environ.copy(), "TAVILY_API_KEY": settings.tavily_api_key},
    }
