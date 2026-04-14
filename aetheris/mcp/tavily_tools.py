"""Tavily MCP server configuration."""
from aetheris.config import get_settings


def get_tavily_server_config() -> dict:
    """Return the Tavily MCP stdio server config for MultiServerMCPClient."""
    settings = get_settings()
    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-tavily"],
        "env": {"TAVILY_API_KEY": settings.tavily_api_key},
    }
