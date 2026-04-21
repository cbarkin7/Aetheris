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
    # Solo escribir si no existe o si le falta el access_token (token de arranque
    # frío que el servidor MCP no ha actualizado aún).
    _write_calendar_token = True
    if _CALENDAR_TOKEN_FILE.exists():
        try:
            existing = json.loads(_CALENDAR_TOKEN_FILE.read_text(encoding="utf-8"))
            # El servidor guarda {"<account>": {access_token, refresh_token, ...}}
            # Si alguna cuenta tiene access_token, el fichero es producto de un auth real.
            has_live_token = any(
                isinstance(v, dict) and v.get("access_token")
                for v in existing.values()
            )
            if has_live_token:
                logger.debug(
                    "[MCP] → ensure_google_token_files | .calendar-token.json ya contiene "
                    "token vivo — no se sobreescribe"
                )
                _write_calendar_token = False
        except Exception:
            pass  # fichero corrupto → sobreescribir

    if _write_calendar_token:
        # Formato mínimo de arranque: @cocal/google-calendar-mcp lo usa para hacer
        # el primer refresh contra GOOGLE_OAUTH_CREDENTIALS.
        calendar_token = {
            "normal": {
                "refresh_token": settings.google_refresh_token,
                "scope": "https://www.googleapis.com/auth/calendar",
                "token_type": "Bearer",
            }
        }
        _CALENDAR_TOKEN_FILE.write_text(json.dumps(calendar_token, indent=2), encoding="utf-8")
        logger.debug("[MCP] → ensure_google_token_files | .calendar-token.json escrito (arranque frío)")

    # ── Gmail ─────────────────────────────────────────────────────────────────
    # Solo escribir si no existe o si le falta el access_token (token Node.js).
    _write_gmail_token = True
    if _GMAIL_TOKEN_FILE.exists():
        try:
            existing = json.loads(_GMAIL_TOKEN_FILE.read_text(encoding="utf-8"))
            # Formato Node.js OAuth2: {access_token, refresh_token, expiry_date, ...}
            if existing.get("access_token") and existing.get("refresh_token"):
                logger.debug(
                    "[MCP] → ensure_google_token_files | .gmail-token.json ya contiene "
                    "token vivo — no se sobreescribe"
                )
                _write_gmail_token = False
        except Exception:
            pass

    if _write_gmail_token:
        # Formato Node.js OAuth2 que espera @gongrzhe/server-gmail-autoauth-mcp.
        # Sin access_token el servidor hará el refresh automáticamente.
        gmail_token = {
            "access_token": "",
            "refresh_token": settings.google_refresh_token,
            "scope": "https://mail.google.com/",
            "token_type": "Bearer",
            "expiry_date": 1000,   # forzar refresh inmediato en el primer uso
        }
        _GMAIL_TOKEN_FILE.write_text(json.dumps(gmail_token, indent=2), encoding="utf-8")
        logger.debug("[MCP] → ensure_google_token_files | .gmail-token.json escrito (arranque frío)")

    logger.debug(
        "[MCP] → ensure_google_token_files | Calendar=%s Gmail=%s",
        _CALENDAR_TOKEN_FILE, _GMAIL_TOKEN_FILE,
    )
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
    Configuración del servidor MCP de Gmail.
    Usa @gongrzhe/server-gmail-autoauth-mcp vía npx.

    Variables de entorno que lee el paquete:
      GMAIL_OAUTH_PATH       → ruta COMPLETA al fichero de client secret (gcp-oauth.keys.json)
      GMAIL_CREDENTIALS_PATH → ruta COMPLETA al fichero de token OAuth2 (credentials.json)

    NOTA: el paquete espera el fichero de client secret con el nombre gcp-oauth.keys.json
    pero acepta cualquier ruta si GMAIL_OAUTH_PATH apunta al fichero directamente.
    """
    settings = get_settings()
    secret_file = str(Path(settings.google_client_secret_file).resolve())
    token_path = str(_GMAIL_TOKEN_FILE.resolve())

    return {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        "env": {
            **os.environ.copy(),
            # client secret → GMAIL_OAUTH_PATH (no GOOGLE_OAUTH_CREDENTIALS)
            "GMAIL_OAUTH_PATH": secret_file,
            # token/credentials → GMAIL_CREDENTIALS_PATH (no CREDENTIALS_PATH)
            "GMAIL_CREDENTIALS_PATH": token_path,
        },
    }


# ---------------------------------------------------------------------------
# Alias de compatibilidad con versiones anteriores
# ---------------------------------------------------------------------------

def ensure_google_credentials_files() -> bool:
    """Alias de ensure_google_token_files() — mantenido por compatibilidad."""
    return ensure_google_token_files()


def get_google_server_config() -> dict:
    """Alias de compatibilidad → calendar_server_config()."""
    return calendar_server_config()
