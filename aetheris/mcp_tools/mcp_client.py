import os
from pathlib import Path


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

    Nota: Gmail usa gmail_mcp_server.py (stdio Python) con su propio par de
    variables (GMAIL_TOKEN_PATH, GMAIL_CLIENT_SECRET_PATH) gestionadas en
    google_tools.gmail_server_config(). No se añaden aquí.
    """
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

    return env
