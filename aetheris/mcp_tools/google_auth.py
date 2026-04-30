import os
import json
import time
import requests
from pathlib import Path

_cached_token = None
_cached_until = 0


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_google_access_token() -> str:
    global _cached_token, _cached_until

    if _cached_token and time.time() < _cached_until - 60:
        return _cached_token

    root = get_project_root()
    secret_file = root / "data" / "google" / "client_secret_aetheris.json"

    data = json.loads(secret_file.read_text(encoding="utf-8"))
    client_data = data.get("installed") or data.get("web")

    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": client_data["client_id"],
            "client_secret": client_data["client_secret"],
            "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN"),
            "grant_type": "refresh_token",
        },
        timeout=20,
    )

    response.raise_for_status()
    payload = response.json()

    _cached_token = payload["access_token"]
    _cached_until = time.time() + int(payload.get("expires_in", 3599))

    return _cached_token