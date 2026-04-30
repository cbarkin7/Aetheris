import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from aetheris.mcp_tools.google_auth import get_google_access_token


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_google_env() -> dict:
    root = get_project_root()

    env = os.environ.copy()
    env["GOOGLE_OAUTH_CREDENTIALS"] = str(
        root / "data" / "google" / "client_secret_aetheris.json"
    )
    env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(
        root / "data" / "google" / ".calendar-token.json"
    )
    env["GOOGLE_DRIVE_MCP_TOKEN_PATH"] = str(
        root / "data" / "google" / ".drive-token.json"
    )

    return env


def get_mcp_client() -> MultiServerMCPClient:
    access_token = get_google_access_token()
    google_env = get_google_env()

    return MultiServerMCPClient(
        {
            "calendar": {
                "transport": "stdio",
                "command": "cmd",
                "args": ["/c", "npx", "-y", "@cocal/google-calendar-mcp"],
                "env": google_env,
            },
            "gmail": {
                "transport": "http",
                "url": os.getenv("GMAIL_MCP_URL", "http://localhost:30000/mcp"),
                "headers": {
                    "Authorization": f"Bearer {access_token}"
                },
            },
            "drive": {
                "transport": "stdio",
                "command": "cmd",
                "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-gdrive"],
                "env": google_env,
            },
        }
    )