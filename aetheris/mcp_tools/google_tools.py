"""
Google Workspace MCP server configuration (Calendar, Gmail, Drive).

Paquetes utilizados:
  - @cocal/google-calendar-mcp              → Google Calendar (stdio)
  - @gongrzhe/server-gmail-mcp             → Gmail (HTTP en localhost:30000)
  - @piotr-agier/google-drive-mcp          → Google Drive (stdio)

Credenciales OAuth2:
  - GOOGLE_OAUTH_CREDENTIALS apunta a data/google/client_secret_aetheris.json,
    el fichero JSON descargado desde Google Cloud Console (tipo Desktop app).
    client_id y client_secret se leen directamente de ese fichero.
  - ensure_google_token_files() escribe .calendar-token.json y .gmail-token.json
  - ensure_google_drive_token_files() escribe .drive-token.json (formato raw OAuth2)
  - start_gmail_mcp_server() arranca @gongrzhe/server-gmail-mcp como proceso HTTP local
"""
import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from aetheris.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas de ficheros de token (se generan en tiempo de ejecución)
# ---------------------------------------------------------------------------
# Ruta absoluta anclada al directorio del módulo, independiente del CWD del
# proceso uvicorn. Path("data/google") sería relativa al CWD y fallaría si
# uvicorn no se lanza desde la raíz del proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CREDENTIALS_DIR = _PROJECT_ROOT / "data" / "google"
_CALENDAR_TOKEN_FILE = _CREDENTIALS_DIR / ".calendar-token.json"
_GMAIL_TOKEN_FILE = _CREDENTIALS_DIR / ".gmail-token.json"
_DRIVE_TOKEN_FILE = _CREDENTIALS_DIR / ".drive-token.json"


def _resolve_secret_file() -> Path:
    """Devuelve la ruta absoluta al client_secret_aetheris.json."""
    settings = get_settings()
    p = Path(settings.google_client_secret_file)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _read_client_credentials() -> tuple[str, str]:
    """
    Lee client_id y client_secret desde client_secret_aetheris.json.
    Si el fichero no existe o falla, devuelve los valores de las env vars.
    """
    settings = get_settings()
    secret_file = _resolve_secret_file()

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


def _fetch_fresh_access_token() -> tuple[str, int]:
    """
    Obtiene un access_token fresco vía google_auth.py y devuelve
    (access_token, expiry_date_ms) donde expiry_date_ms es en milisegundos
    (formato que usan los servidores MCP de Google).

    Devuelve ("", 1) si el refresco falla, de modo que el llamador puede
    decidir si usar ese valor o abortar.
    """
    import time as _t
    try:
        from aetheris.mcp_tools.google_auth import get_google_access_token
        access_token = get_google_access_token()
        # Los tokens de Google duran 3600 s; usamos 3540 para no llegar al límite
        expiry_ms = int((_t.time() + 3540) * 1000)
        logger.debug(
            "[MCP] → _fetch_fresh_access_token | access_token obtenido | "
            "expiry_ms=%d", expiry_ms,
        )
        return access_token, expiry_ms
    except Exception as exc:
        logger.warning(
            "[MCP] → _fetch_fresh_access_token | no se pudo obtener access_token: %s "
            "— los servidores MCP deberán refrescar por su cuenta", exc,
        )
        return "", 1


def ensure_google_token_files() -> bool:
    """
    Prepara los ficheros de token para Calendar y Gmail.

    Política de escritura:
    - Siempre actualiza el refresh_token con el valor de .env.
    - Pre-inyecta un access_token real obtenido vía google_auth.py para que
      los servidores MCP no necesiten hacer el refresco en frío (lo que en
      @cocal/google-calendar-mcp provoca el error -32600 «tokens no longer valid»).
    - Si ya existe un access_token válido con el mismo refresh_token, se reutiliza
      sin hacer una nueva llamada a Google.

    Devuelve True si los ficheros están disponibles, False en caso contrario.
    """
    import time as _time
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

    # ── Obtener access_token fresco una sola vez para Calendar y Gmail ────────
    # Se hace aquí, antes de escribir los ficheros, para evitar dos llamadas a
    # Google. Si falla, los servidores arrancan con token vacío y lo refrescarán
    # ellos mismos (comportamiento anterior).
    now_ms = int(_time.time() * 1000)
    fresh_access_token = ""
    fresh_expiry_ms = 1

    # ── Calendar ──────────────────────────────────────────────────────────────
    # Se escribe con dos alias: "normal" (nombre técnico del servidor) y "personal"
    # (nombre natural que el LLM tiende a usar). Ambos apuntan a las mismas credenciales.
    # Así list-events funciona tanto con account="normal" como account="personal".
    _CALENDAR_ACCOUNTS = ("normal", "personal")

    # Determinar qué access_token usar
    reuse_at = ""
    reuse_exp = 1

    if _CALENDAR_TOKEN_FILE.exists():
        try:
            existing = json.loads(_CALENDAR_TOKEN_FILE.read_text(encoding="utf-8"))
            # Leer desde cualquiera de los dos alias (el primero que tenga datos)
            for _acct in _CALENDAR_ACCOUNTS:
                _acct_data = existing.get(_acct, {})
                _at = _acct_data.get("access_token", "")
                _exp = _acct_data.get("expiry_date", 0)
                _same_rt = _acct_data.get("refresh_token") == settings.google_refresh_token
                if _at and _exp > now_ms + 60_000 and _same_rt:
                    reuse_at = _at
                    reuse_exp = _exp
                    logger.debug(
                        "[MCP] → ensure_google_token_files | .calendar-token.json: "
                        "token vigente en cuenta '%s' — reutilizando", _acct,
                    )
                    break
        except Exception:
            pass  # fichero corrupto → token limpio

    if not reuse_at:
        # Token expirado, vacío o fichero nuevo → obtener token fresco
        if not fresh_access_token:
            fresh_access_token, fresh_expiry_ms = _fetch_fresh_access_token()
        reuse_at = fresh_access_token
        reuse_exp = fresh_expiry_ms
        logger.debug(
            "[MCP] → ensure_google_token_files | .calendar-token.json: "
            "usando access_token fresco"
        )

    _account_entry = {
        "access_token": reuse_at,
        "refresh_token": settings.google_refresh_token,
        "scope": "https://www.googleapis.com/auth/calendar",
        "token_type": "Bearer",
        "expiry_date": reuse_exp,
    }
    calendar_token = {acct: _account_entry for acct in _CALENDAR_ACCOUNTS}

    _CALENDAR_TOKEN_FILE.write_text(json.dumps(calendar_token, indent=2), encoding="utf-8")
    logger.debug("[MCP] → ensure_google_token_files | .calendar-token.json escrito (cuentas: %s)",
                 list(_CALENDAR_ACCOUNTS))

    # ── Gmail ─────────────────────────────────────────────────────────────────
    # Misma política que Calendar: pre-inyectar access_token fresco si es necesario.
    # Reutilizar el ya obtenido para Calendar (misma credencial, mismo token).
    gmail_token: dict = {
        "access_token": "",
        "refresh_token": settings.google_refresh_token,
        "scope": "https://mail.google.com/",
        "token_type": "Bearer",
        "expiry_date": 1,
    }
    if _GMAIL_TOKEN_FILE.exists():
        try:
            existing = json.loads(_GMAIL_TOKEN_FILE.read_text(encoding="utf-8"))
            existing_at = existing.get("access_token", "")
            existing_exp = existing.get("expiry_date", 0)
            same_rt = existing.get("refresh_token") == settings.google_refresh_token
            token_still_valid = existing_at and existing_exp > now_ms + 60_000 and same_rt

            if token_still_valid:
                gmail_token["access_token"] = existing_at
                gmail_token["expiry_date"] = existing_exp
                logger.debug(
                    "[MCP] → ensure_google_token_files | .gmail-token.json: "
                    "token existente vigente — reutilizando"
                )
            else:
                # Reutilizar el fresh_access_token ya obtenido para Calendar
                if not fresh_access_token:
                    fresh_access_token, fresh_expiry_ms = _fetch_fresh_access_token()
                gmail_token["access_token"] = fresh_access_token
                gmail_token["expiry_date"] = fresh_expiry_ms
                logger.debug(
                    "[MCP] → ensure_google_token_files | .gmail-token.json: "
                    "token expirado/cambiado — usando access_token fresco"
                )
        except Exception:
            pass
    else:
        if not fresh_access_token:
            fresh_access_token, fresh_expiry_ms = _fetch_fresh_access_token()
        gmail_token["access_token"] = fresh_access_token
        gmail_token["expiry_date"] = fresh_expiry_ms

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

    @piotr-agier/google-drive-mcp usa @google-cloud/local-auth, que guarda y lee
    tokens en formato 'authorized_user':
        {
          "type": "authorized_user",
          "client_id": "...",
          "client_secret": "...",
          "refresh_token": "..."
        }

    Este formato es el que google.auth.fromJSON() acepta en @google-cloud/local-auth.
    El servidor renueva el access_token automáticamente usando el refresh_token.

    Política de escritura: siempre actualizar con el refresh_token actual de .env.
    Si el fichero ya existe en formato authorized_user con el mismo refresh_token,
    se reutiliza sin sobreescribir (el servidor podría haberlo actualizado).

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

    # Comprobar si ya existe un fichero authorized_user válido con el mismo refresh_token
    if _DRIVE_TOKEN_FILE.exists():
        try:
            existing = json.loads(_DRIVE_TOKEN_FILE.read_text(encoding="utf-8"))
            if (
                existing.get("type") == "authorized_user"
                and existing.get("refresh_token") == settings.google_refresh_token
            ):
                logger.debug(
                    "[MCP] → ensure_google_drive_token_files | .drive-token.json: "
                    "authorized_user válido con refresh_token coincidente — reutilizando"
                )
                return True
        except Exception:
            pass  # fichero corrupto → reescribir

    # Escribir en formato authorized_user (requerido por @google-cloud/local-auth)
    drive_token: dict = {
        "type": "authorized_user",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": settings.google_refresh_token,
    }

    _DRIVE_TOKEN_FILE.write_text(json.dumps(drive_token, indent=2), encoding="utf-8")
    logger.info(
        "[MCP] → ensure_google_drive_token_files | .drive-token.json escrito "
        "(formato authorized_user para @google-cloud/local-auth)"
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
    GOOGLE_CALENDAR_MCP_TOKEN_PATH apunta al token authorized_user.

    En Windows se usa `cmd /c npx`; en Linux/Mac se usa `npx` directamente.
    """
    secret_file = str(_resolve_secret_file())
    token_path = str(_CALENDAR_TOKEN_FILE)

    if os.name == "nt":
        cmd, args_prefix = "cmd", ["/c", "npx", "-y"]
    else:
        cmd, args_prefix = "npx", ["-y"]

    return {
        "transport": "stdio",
        "command": cmd,
        "args": args_prefix + ["@cocal/google-calendar-mcp"],
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
    Usa @piotr-agier/google-drive-mcp vía npx.

    Herramientas disponibles: search, listFolder, uploadFile, downloadFile,
    createTextFile, updateTextFile, deleteItem, moveItem, renameItem, copyFile,
    createGoogleDoc/Sheet/Slides, operaciones Docs/Sheets/Slides, y Calendar.

    Variables de entorno requeridas:
    - GOOGLE_DRIVE_OAUTH_CREDENTIALS → client_secret JSON (flujo OAuth2)
    - GOOGLE_DRIVE_MCP_TOKEN_PATH    → token en formato authorized_user

    En Windows se usa `cmd /c npx`; en Linux/Mac se usa `npx` directamente.
    """
    secret_file = str(_resolve_secret_file())
    token_path = str(_DRIVE_TOKEN_FILE)

    if os.name == "nt":
        cmd, args_prefix = "cmd", ["/c", "npx", "-y"]
    else:
        cmd, args_prefix = "npx", ["-y"]

    return {
        "transport": "stdio",
        "command": cmd,
        "args": args_prefix + ["@piotr-agier/google-drive-mcp"],
        "env": {
            **os.environ.copy(),
            "GOOGLE_DRIVE_OAUTH_CREDENTIALS": secret_file,
            "GOOGLE_DRIVE_MCP_TOKEN_PATH": token_path,
        },
    }


# ---------------------------------------------------------------------------
# Gmail MCP HTTP server — gestión del proceso local
# ---------------------------------------------------------------------------

async def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """
    Espera (async) hasta que el puerto TCP esté aceptando conexiones.
    Devuelve True si el puerto abrió antes del timeout, False si no.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


async def start_gmail_mcp_server() -> "subprocess.Popen | None":
    """
    Arranca @gongrzhe/server-gmail-mcp como proceso HTTP local y espera a que
    el puerto esté disponible antes de devolver el control.

    Solo se arranca si:
    - GMAIL_MCP_URL apunta a localhost / 127.0.0.1 (servidor externo → no tocar)
    - GOOGLE_REFRESH_TOKEN está configurado

    El proceso devuelto debe guardarse en app.state.gmail_process y terminarse
    en el shutdown del lifespan de FastAPI.

    Returns: subprocess.Popen activo, o None si no aplica o falla el arranque.
    """
    settings = get_settings()

    # Solo autoarrancar si la URL apunta a localhost
    parsed = urlparse(settings.gmail_mcp_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 30000
    if host not in ("localhost", "127.0.0.1"):
        logger.info(
            "[MCP] → start_gmail_mcp_server | GMAIL_MCP_URL es externo (%s) — no autoarrancando",
            settings.gmail_mcp_url,
        )
        return None

    if not settings.google_refresh_token:
        logger.warning(
            "[MCP] → start_gmail_mcp_server | GOOGLE_REFRESH_TOKEN no configurado — omitiendo"
        )
        return None

    secret_file = str(_resolve_secret_file())
    token_path = str(_GMAIL_TOKEN_FILE)

    # El servidor Gmail MCP necesita las mismas credenciales que Calendar
    gmail_env = {
        **os.environ.copy(),
        "GOOGLE_OAUTH_CREDENTIALS": secret_file,
        "GOOGLE_GMAIL_MCP_TOKEN_PATH": token_path,
    }

    if os.name == "nt":
        cmd = ["cmd", "/c", "npx", "-y", "@gongrzhe/server-gmail-mcp"]
    else:
        cmd = ["npx", "-y", "@gongrzhe/server-gmail-mcp"]

    try:
        logger.info(
            "[MCP] → start_gmail_mcp_server | arrancando servidor Gmail MCP HTTP | "
            "cmd=%s token=%s",
            cmd, token_path,
        )
        process = subprocess.Popen(
            cmd,
            env=gmail_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        logger.info(
            "[MCP] → start_gmail_mcp_server | proceso arrancado pid=%d | "
            "esperando puerto %s:%d (timeout=30s)",
            process.pid, host, port,
        )
        ready = await _wait_for_port(host, port, timeout=30.0)

        if ready:
            logger.info(
                "[MCP] → start_gmail_mcp_server | servidor Gmail MCP listo | "
                "pid=%d url=%s",
                process.pid, settings.gmail_mcp_url,
            )
        else:
            logger.warning(
                "[MCP] → start_gmail_mcp_server | timeout esperando puerto %d — "
                "Gmail MCP podría no estar listo",
                port,
            )

        return process

    except Exception as exc:
        logger.error(
            "[MCP] → start_gmail_mcp_server | error arrancando servidor: %s", exc
        )
        return None


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
