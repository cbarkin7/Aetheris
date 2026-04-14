"""Google Workspace MCP server configuration (Calendar, Gmail, Drive)."""
from aetheris.config import get_settings


def get_google_server_config() -> dict:
    """Return the Google MCP stdio server config for MultiServerMCPClient."""
    settings = get_settings()
    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@googleapis/mcp-server-google"],
        "env": {
            "GOOGLE_CLIENT_ID": settings.google_client_id,
            "GOOGLE_CLIENT_SECRET": settings.google_client_secret,
            "GOOGLE_REFRESH_TOKEN": settings.google_refresh_token,
        },
    }
