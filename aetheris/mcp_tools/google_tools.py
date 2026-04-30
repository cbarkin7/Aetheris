"""
Google Workspace MCP server configuration (Calendar, Gmail).

Paquetes utilizados:
  - @cocal/google-calendar-mcp              → Google Calendar
  - @gongrzhe/server-gmail-autoauth-mcp     → Gmail

Credenciales OAuth2:
  - GOOGLE_OAUTH_CREDENTIALS apunta a data/google/client_secret_aetheris.json,
    el fichero JSON descargado desde Google Cloud Console (tipo Desktop app).
    client_id y client_secret se leen directamente de ese fichero.
  - ensure_google_token_files() escribe los ficheros de token authorized_user
    (.calendar-token.json / .gmail-token.json) a partir de GOOGLE_REFRESH_TOKEN,
    evitando el flujo OAuth interactivo cuando ya se dispone de un refresh_token válido.
"""
import json
import logging
import os
from pathlib import Path

from aetheris.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas de ficheros de token (se generan en tiempo de ejecución)
# ---------------------------------------------------------------------------
_CREDENTIALS_DIR = Path("data/google")
_CALENDAR_TOKEN_FILE = _CREDENTIALS_DIR / ".calendar-token.json"
_GMAIL_TOKEN_FILE = _CREDENTIALS_DIR / ".gmail-token.json"
_DRIVE_TOKEN_FILE = _CREDENTIALS_DIR / ".drive-token.json"


def _read_client_credentials() -> tuple[str, str]:
    """
    Lee client_id y client_secret desde client_secret_aetheris.json.
    Si el fichero no existe o falla, devuelve los valores de las env vars.
    """
    settings = get_settings()
    secret_file = Path(settings.google_client_secret_file)

    if secret_file.exists():
        try:
            data = json.loads(secret_file.read_text(encoding="utf-8"))
            # Formato Google Cloud Console: {"installed": {...}} o {"web": {...}}
            inner = data.get("installed") or data.get("web") or {}
            client_id = inner.get("client_id", settings.google_client_id)
            client_secret = inner.get("client_secret", settings.google_client_secret)
            return client_id, client_secret
        except Exception as exc:
            logger.warning("[MCP] → _read_client_credentials | error leyendo %s: %s", secret_file, exc)

    return settings.google_client_id, settings.google_client_secret


def ensure_google_token_files() -> bool:
    """
    Prepara los ficheros de token para Calendar y Gmail.

    Política de escritura:
    - Si el fichero ya existe con un token «vivo» generado por el propio servidor MCP
      (contiene access_token o la estructura nativa del paquete), NO se sobreescribe:
      los tokens MCP incluyen access_token + expiry_date que el servidor sabe refrescar.
    - Si el fichero no existe o solo tiene el esqueleto mínimo (solo refresh_token),
      se escribe la versión de arranque en frío que permite el primer refresh.

    Devuelve True si los ficheros están disponibles, False en caso contrario.
    """
    settings = get_settings()

    if not settings.google_refresh_token:
        logger.warning(
            "[MCP] → ensure_google_token_files | GOOGLE_REFRESH_TOKEN no configurado"
        )
        return False

    client_id, client_secret = _read_client_credentials()
    if not (client_id and client_secret):
        logger.warning(
            "[MCP] → ensure_google_token_files | client_id / client_secret no disponibles"
        )
        return False

    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Calendar ──────────────────────────────────────────────────────────────
    # Política: siempre actualizar el refresh_token con el valor actual de .env.
    # Preservar el access_token existente si está presente (evita un refresh innecesario),
    # pero NUNCA conservar un refresh_token distinto al de .env (podría estar revocado).
    calendar_token: dict = {
        "normal": {
            "access_token": "",
            "refresh_token": settings.google_refresh_token,
            "scope": "https://www.googleapis.com/auth/calendar",
            "token_type": "Bearer",
            "expiry_date": 1,   # forzar refresh inmediato — el servidor MCP usará el refresh_token
        }
    }
    if _CALENDAR_TOKEN_FILE.exists():
        try:
            existing = json.loads(_CALENDAR_TOKEN_FILE.read_text(encoding="utf-8"))
            normal = existing.get("normal", {})
            # Reutilizar access_token solo si el refresh_token coincide con el actual
            if (
                isinstance(normal, dict)
                and normal.get("access_token")
                and normal.get("expiry_date", 0) > 1   # > 1 significa token real, no cold-start
                and normal.get("refresh_token") == settings.google_refresh_token
            ):
                calendar_token["normal"]["access_token"] = normal["access_token"]
                calendar_token["normal"]["expiry_date"] = normal["expiry_date"]
                logger.debug(
                    "[MCP] → ensure_google_token_files | .calendar-token.json: "
                    "refresh_token coincide — reutilizando access_token existente"
                )
            else:
                logger.debug(
                    "[MCP] → ensure_google_token_files | .calendar-token.json: "
                    "refresh_token cambiado o sin access_token — escribiendo token limpio"
                )
        except Exception:
            pass  # fichero corrupto → usar token limpio

    _CALENDAR_TOKEN_FILE.write_text(json.dumps(calendar_token, indent=2), encoding="utf-8")
    logger.debug("[MCP] → ensure_google_token_files | .calendar-token.json escrito")

    # ── Gmail ─────────────────────────────────────────────────────────────────
    # Misma política: siempre actualizar el refresh_token con el valor de .env.
    gmail_token: dict = {
        "access_token": "",
        "refresh_token": settings.google_refresh_token,
        "scope": "https://mail.google.com/",
        "token_type": "Bearer",
        "expiry_date": 1000,   # forzar refresh inmediato en el primer uso
    }
    if _GMAIL_TOKEN_FILE.exists():
        try:
            existing = json.loads(_GMAIL_TOKEN_FILE.read_text(encoding="utf-8"))
            # Reutilizar access_token solo si el refresh_token coincide con el actual
            if (
                existing.get("access_token")
                and existing.get("refresh_token") == settings.google_refresh_token
            ):
                gmail_token["access_token"] = existing["access_token"]
                if existing.get("expiry_date"):
                    gmail_token["expiry_date"] = existing["expiry_date"]
                logger.debug(
                    "[MCP] → ensure_google_token_files | .gmail-token.json: "
                    "refresh_token coincide — reutilizando access_token existente"
                )
            else:
                logger.debug(
                    "[MCP] → ensure_google_token_files | .gmail-token.json: "
                    "refresh_token cambiado o sin access_token — escribiendo token limpio"
                )
        except Exception:
            pass

    _GMAIL_TOKEN_FILE.write_text(json.dumps(gmail_token, indent=2), encoding="utf-8")
    logger.debug("[MCP] → ensure_google_token_files | .gmail-token.json escrito")

    logger.debug(
        "[MCP] → ensure_google_token_files | Calendar=%s Gmail=%s",
        _CALENDAR_TOKEN_FILE, _GMAIL_TOKEN_FILE,
    )
    return True


def ensure_google_drive_token_files() -> bool:
    """
    Prepara el fichero de token para Google Drive.

    Misma política que Calendar: siempre actualizar el refresh_token con el valor
    de .env; preservar el access_token si el refresh_token coincide y expiry_date > 1.

    Devuelve True si el fichero está disponible, False en caso contrario.
    """
    settings = get_settings()

    if not settings.google_refresh_token:
        logger.warning(
            "[MCP] → ensure_google_drive_token_files | GOOGLE_REFRESH_TOKEN no configurado"
        )
        return False

    client_id, client_secret = _read_client_credentials()
    if not (client_id and client_secret):
        logger.warning(
            "[MCP] → ensure_google_drive_token_files | client_id / client_secret no disponibles"
        )
        return False

    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    drive_token: dict = {
        "access_token": "",
        "refresh_token": settings.google_refresh_token,
        "scope": "https://www.googleapis.com/auth/drive",
        "token_type": "Bearer",
        "expiry_date": 1,  # forzar refresh inmediato en el primer uso
    }

    if _DRIVE_TOKEN_FILE.exists():
        try:
            existing = json.loads(_DRIVE_TOKEN_FILE.read_text(encoding="utf-8"))
            if (
                existing.get("access_token")
                and existing.get("expiry_date", 0) > 1
                and existing.get("refresh_token") == settings.google_refresh_token
            ):
                drive_token["access_token"] = existing["access_token"]
                drive_token["expiry_date"] = existing["expiry_date"]
                logger.debug(
                    "[MCP] → ensure_google_drive_token_files | .drive-token.json: "
                    "refresh_token coincide — reutilizando access_token existente"
                )
            else:
                logger.debug(
                    "[MCP] → ensure_google_drive_token_files | .drive-token.json: "
                    "refresh_token cambiado o sin access_token — escribiendo token limpio"
                )
        except Exception:
            pass  # fichero corrupto → usar token limpio

    _DRIVE_TOKEN_FILE.write_text(json.dumps(drive_token, indent=2), encoding="utf-8")
    logger.debug("[MCP] → ensure_google_drive_token_files | .drive-token.json escrito")
    return True


# ---------------------------------------------------------------------------
# Configuraciones de servidor MCP
# ---------------------------------------------------------------------------

def calendar_server_config() -> dict:
    """
    Configuración del servidor MCP de Google Calendar.
    Usa @cocal/google-calendar-mcp vía npx.

    GOOGLE_OAUTH_CREDENTIALS apunta al fichero client_secret_aetheris.json.
    GOOGLE_TOKEN_PATH / GOOGLE_CALENDAR_TOKEN apuntan al token authorized_user.
    """
    settings = get_settings()
    secret_file = str(Path(settings.google_client_secret_file).resolve())
    token_path = str(_CALENDAR_TOKEN_FILE.resolve())

    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@cocal/google-calendar-mcp"],
        "env": {
            **os.environ.copy(),
            "GOOGLE_OAUTH_CREDENTIALS": secret_file,
            # Ruta personalizada del token — evita usar C:\Users\barki\.config\google-calendar-mcp\tokens.json
            "GOOGLE_CALENDAR_MCP_TOKEN_PATH": token_path,
        },
    }


def gmail_server_config() -> dict:
    """
    Configuración del servidor MCP de Gmail (HTTP + Bearer token).

    Requiere un servidor Gmail MCP HTTP corriendo en GMAIL_MCP_URL
    (p.ej. Composio o servidor auto-hospedado).

    El access_token tiene validez ~1 hora. Limitación conocida del TFM.
    """
    from aetheris.mcp_tools.google_auth import get_google_access_token
    settings = get_settings()
    access_token = get_google_access_token()

    return {
        "transport": "http",
        "url": settings.gmail_mcp_url,
        "headers": {"Authorization": f"Bearer {access_token}"},
    }


def drive_server_config() -> dict:
    """
    Configuración del servidor MCP de Google Drive.
    Usa @modelcontextprotocol/server-gdrive vía npx.

    GOOGLE_OAUTH_CREDENTIALS apunta al fichero client_secret_aetheris.json.
    GOOGLE_DRIVE_MCP_TOKEN_PATH apunta al token authorized_user para Drive.

    En Windows se usa `cmd /c npx`; en Linux/Mac se usa `npx` directamente.
    """
    settings = get_settings()
    secret_file = str(Path(settings.google_client_secret_file).resolve())
    token_path = str(_DRIVE_TOKEN_FILE.resolve())

    if os.name == "nt":
        cmd, args_prefix = "cmd", ["/c", "npx", "-y"]
    else:
        cmd, args_prefix = "npx", ["-y"]

    return {
        "transport": "stdio",
        "command": cmd,
        "args": args_prefix + ["@modelcontextprotocol/server-gdrive"],
        "env": {
            **os.environ.copy(),
            "GOOGLE_OAUTH_CREDENTIALS": secret_file,
            "GOOGLE_DRIVE_MCP_TOKEN_PATH": token_path,
        },
    }


# ---------------------------------------------------------------------------
# Alias de compatibilidad con versiones anteriores
# ---------------------------------------------------------------------------

def ensure_google_credentials_files() -> bool:
    """Prepara los ficheros de token para Calendar, Gmail y Drive."""
    calendar_ok = ensure_google_token_files()
    drive_ok = ensure_google_drive_token_files()
    return calendar_ok and drive_ok


def get_google_server_config() -> dict:
    """Alias de compatibilidad → calendar_server_config()."""
    return calendar_server_config()
