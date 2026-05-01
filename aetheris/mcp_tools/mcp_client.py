import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from aetheris.mcp_tools.google_auth import get_google_access_token


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_google_env() -> dict:
    """
    Construye el dict de variables de entorno para los servidores MCP de Google.

    Variables clave:
    - GOOGLE_OAUTH_CREDENTIALS         → client_secret JSON para @cocal/google-calendar-mcp
    - GOOGLE_CALENDAR_MCP_TOKEN_PATH   → token Calendar (formato {"normal": {...}})
    - GOOGLE_DRIVE_OAUTH_CREDENTIALS   → client_secret JSON para @piotr-agier/google-drive-mcp
    - GOOGLE_DRIVE_MCP_TOKEN_PATH      → token Drive (formato authorized_user)
    - GOOGLE_GMAIL_MCP_TOKEN_PATH      → token Gmail para @gongrzhe/server-gmail-mcp
    """
    import json
    root = get_project_root()
    secret_file = root / "data" / "google" / "client_secret_aetheris.json"

    env = os.environ.copy()
    env["GOOGLE_OAUTH_CREDENTIALS"] = str(secret_file)
    env["GOOGLE_CALENDAR_MCP_TOKEN_PATH"] = str(
        root / "data" / "google" / ".calendar-token.json"
    )
    # @piotr-agier/google-drive-mcp usa:
    #   GOOGLE_DRIVE_OAUTH_CREDENTIALS → client_secret JSON (para el flujo OAuth2)
    #   GOOGLE_DRIVE_MCP_TOKEN_PATH    → token guardado en formato authorized_user
    env["GOOGLE_DRIVE_OAUTH_CREDENTIALS"] = str(secret_file)
    env["GOOGLE_DRIVE_MCP_TOKEN_PATH"] = str(
        root / "data" / "google" / ".drive-token.json"
    )
    env["GOOGLE_GMAIL_MCP_TOKEN_PATH"] = str(
        root / "data" / "google" / ".gmail-token.json"
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
                "args": ["/c", "npx", "-y", "@piotr-agier/google-drive-mcp"],
                "env": google_env,
            },
        }
    )