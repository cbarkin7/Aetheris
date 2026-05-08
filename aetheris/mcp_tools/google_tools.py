"""
Google Workspace MCP server configuration (Calendar, Gmail, Drive).

Paquetes y servidores utilizados:
  - @cocal/google-calendar-mcp              → Google Calendar (stdio, npx)
  - gmail_mcp_server.py                     → Gmail (stdio, Python nativo, google-auth)
  - @piotr-agier/google-drive-mcp          → Google Drive (stdio, npx)

Credenciales OAuth2:
  - GOOGLE_OAUTH_CREDENTIALS apunta a data/google/client_secret_aetheris.json,
    el fichero JSON descargado desde Google Cloud Console (tipo Desktop app).
    client_id y client_secret se leen directamente de ese fichero.
  - ensure_google_token_files() escribe .calendar-token.json y .gmail-token.json
  - ensure_google_drive_token_files() escribe .drive-token.json (formato authorized_user)
  - gmail_server_config() lanza gmail_mcp_server.py vía stdio con GMAIL_TOKEN_PATH/
    GMAIL_CLIENT_SECRET_PATH; el servidor refresca el token con google-auth internamente.
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
    # Pre-inyectar access_token fresco en .gmail-token.json para que
    # gmail_mcp_server.py no tenga que hacer el primer refresco en frío.
    # Reutiliza el token ya obtenido para Calendar (misma credencial OAuth2).
    # Los scopes son los requeridos por la Gmail REST API (modify + readonly + send).
    gmail_token: dict = {
        "access_token": "",
        "refresh_token": settings.google_refresh_token,
        "scope": (
            "https://www.googleapis.com/auth/gmail.modify "
            "https://www.googleapis.com/auth/gmail.readonly "
            "https://www.googleapis.com/auth/gmail.send"
        ),
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


def _refresh_drive_access_token(client_id: str, client_secret: str, refresh_token: str) -> dict | None:
    """
    Solicita un access_token fresco a Google OAuth2 para Drive.
    Devuelve dict con access_token, expiry_date y token_uri, o None si falla.
    """
    import time
    try:
        import requests as _requests
        resp = _requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            },
            timeout=15,
        )
        if resp.ok:
            body = resp.json()
            expires_in = body.get("expires_in", 3599)
            return {
                "access_token": body["access_token"],
                "expiry_date": int((time.time() + expires_in) * 1000),
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        logger.warning(
            "[MCP] → _refresh_drive_access_token | fallo HTTP %s: %s",
            resp.status_code, resp.text[:200],
        )
    except Exception as exc:
        logger.warning("[MCP] → _refresh_drive_access_token | excepcion: %s", exc)
    return None


def ensure_google_drive_token_files() -> bool:
    """
    Prepara el fichero de token para Google Drive con access_token activo.

    @piotr-agier/google-drive-mcp requiere el token en formato 'authorized_user'
    con access_token y token_uri para NO activar el flujo OAuth interactivo:
        {
          "type": "authorized_user",
          "client_id": "...",
          "client_secret": "...",
          "refresh_token": "...",
          "access_token": "...",     ← requerido para omitir el flujo OAuth
          "expiry_date": 1234567890,
          "token_uri": "https://oauth2.googleapis.com/token"
        }

    Política de escritura:
    - Si el fichero tiene access_token activo y refresh_token coincide → reutilizar.
    - Si el fichero no tiene access_token o está caducado → refrescar y sobrescribir.
    - Si el refresco falla → escribir sin access_token (el servidor intentará su propio flujo).

    Devuelve True si el fichero está disponible, False en caso contrario.
    """
    import time as _time
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

    # Comprobar si ya existe un fichero válido con access_token activo
    if _DRIVE_TOKEN_FILE.exists():
        try:
            existing = json.loads(_DRIVE_TOKEN_FILE.read_text(encoding="utf-8"))
            rt_matches = existing.get("refresh_token") == settings.google_refresh_token
            has_access = bool(existing.get("access_token"))
            # expiry_date está en milisegundos; dejar margen de 5 minutos
            expiry_ms = existing.get("expiry_date", 0)
            is_fresh = expiry_ms > (_time.time() + 300) * 1000
            if existing.get("type") == "authorized_user" and rt_matches and has_access and is_fresh:
                logger.debug(
                    "[MCP] → ensure_google_drive_token_files | .drive-token.json: "
                    "access_token activo con refresh_token coincidente — reutilizando"
                )
                return True
        except Exception:
            pass  # fichero corrupto → reescribir

    # Refrescar access_token para evitar el flujo OAuth interactivo
    token_extra = _refresh_drive_access_token(client_id, client_secret, settings.google_refresh_token)
    if token_extra:
        logger.info(
            "[MCP] → ensure_google_drive_token_files | access_token obtenido (expira en ~%ds)",
            (token_extra["expiry_date"] // 1000 - int(_time.time())),
        )
    else:
        logger.warning(
            "[MCP] → ensure_google_drive_token_files | no se pudo obtener access_token; "
            "el servidor Drive puede solicitar autorizacion interactiva"
        )
        token_extra = {}

    drive_token: dict = {
        "type": "authorized_user",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": settings.google_refresh_token,
        **token_extra,
    }

    _DRIVE_TOKEN_FILE.write_text(json.dumps(drive_token, indent=2), encoding="utf-8")
    logger.info(
        "[MCP] → ensure_google_drive_token_files | .drive-token.json escrito "
        "(formato authorized_user con%s access_token)",
        "" if token_extra.get("access_token") else " SIN",
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


def _ensure_gmail_token_has_client_credentials() -> None:
    """
    Asegura que .gmail-token.json incluye client_id y client_secret.

    google-auth necesita client_id + client_secret para refrescar el
    access_token automáticamente. Si el fichero solo tiene access_token y
    refresh_token (formato minimal), los añade desde client_secret_aetheris.json.
    """
    if not _GMAIL_TOKEN_FILE.exists():
        return
    try:
        token_data = json.loads(_GMAIL_TOKEN_FILE.read_text(encoding="utf-8"))
        if token_data.get("client_id") and token_data.get("client_secret"):
            return  # Ya están presentes
        client_id, client_secret = _read_client_credentials()
        if not (client_id and client_secret):
            return
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret
        _GMAIL_TOKEN_FILE.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
        logger.debug(
            "[MCP] → _ensure_gmail_token_has_client_credentials | "
            "client_id/secret añadidos a .gmail-token.json"
        )
    except Exception as exc:
        logger.warning(
            "[MCP] → _ensure_gmail_token_has_client_credentials | error: %s", exc
        )


def gmail_server_config() -> dict:
    """
    Configuración del servidor MCP de Gmail — Python nativo, transporte stdio.

    Servidor MCP Python propio (aetheris/mcp_tools/gmail_mcp_server.py) que
    usa google-auth y la Gmail REST API directamente vía transporte stdio.

    Ventajas sobre el anterior servidor npm HTTP:
    - Sin dependencia de npm ni proceso HTTP externo
    - Refresh de token OAuth2 automático y robusto (google-auth nativo)
    - Mismo transporte stdio que Calendar y Drive → arquitectura uniforme
    - Producción-ready: no hay paths hardcodeados en ~/.gmail-mcp/
    """
    import sys

    _ensure_gmail_token_has_client_credentials()
    server_script = str(Path(__file__).parent / "gmail_mcp_server.py")
    secret_file = str(_resolve_secret_file())
    token_path = str(_GMAIL_TOKEN_FILE.resolve())

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [server_script],
        "env": {
            **os.environ.copy(),
            "GMAIL_TOKEN_PATH": token_path,
            "GMAIL_CLIENT_SECRET_PATH": secret_file,
        },
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
# Inicialización de credenciales (punto de entrada desde client.py)
# ---------------------------------------------------------------------------

def ensure_google_credentials_files() -> bool:
    """Prepara los ficheros de token para Calendar, Gmail y Drive."""
    calendar_ok = ensure_google_token_files()
    drive_ok = ensure_google_drive_token_files()
    return calendar_ok and drive_ok
