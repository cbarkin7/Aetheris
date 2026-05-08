import logging
import json
import time
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

_cached_token = None
_cached_until = 0


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_google_access_token() -> str:
    """
    Obtiene un access token OAuth2 de Google usando el refresh token.
    Usa get_settings() (pydantic-settings) para leer las credenciales desde
    .env, a diferencia de os.getenv() que solo lee variables del OS.
    """
    global _cached_token, _cached_until

    if _cached_token and time.time() < _cached_until - 60:
        logger.debug("[AUTH] → get_google_access_token | usando token en caché")
        return _cached_token

    # Importar aquí para evitar importaciones circulares en tiempo de módulo
    from aetheris.config import get_settings
    settings = get_settings()

    root = get_project_root()
    secret_file = root / "data" / "google" / "client_secret_aetheris.json"

    data = json.loads(secret_file.read_text(encoding="utf-8"))
    client_data = data.get("installed") or data.get("web")

    # Usar settings.google_refresh_token (pydantic lee .env).
    # os.getenv("GOOGLE_REFRESH_TOKEN") puede devolver None si la variable
    # no está exportada al entorno del OS, aunque esté definida en .env.
    refresh_token = settings.google_refresh_token

    payload = {
        "client_id": client_data["client_id"],
        "client_secret": client_data["client_secret"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data=payload,
        timeout=20,
    )
    response.raise_for_status()
    resp_payload = response.json()

    _cached_token = resp_payload["access_token"]
    _cached_until = time.time() + int(resp_payload.get("expires_in", 3599))

    logger.debug("[AUTH] → get_google_access_token | token obtenido | expira_en=%ss",
                 resp_payload.get("expires_in"))
    return _cached_token
