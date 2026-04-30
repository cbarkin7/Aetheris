from pathlib import Path
from urllib.parse import urlencode
from dotenv import load_dotenv
import json
import os
import requests

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILE = ROOT / "data" / "google" / "client_secret_aetheris.json"

TOKEN_OUTPUT = ROOT / "data" / "google" / "google-token-all.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

data = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
client_data = data.get("installed") or data.get("web")

client_id = client_data["client_id"]
client_secret = client_data["client_secret"]

redirect_uri = "http://localhost"

params = {
    "client_id": client_id,
    "redirect_uri": redirect_uri,
    "response_type": "code",
    "scope": " ".join(SCOPES),
    "access_type": "offline",
    "prompt": "consent",
}

auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)

print("\n1) Abre esta URL en el navegador:\n")
print(auth_url)

print("\n2) Acepta permisos.")
print("3) Te redirigirá a una URL tipo:")
print("   http://localhost/?code=XXXX&scope=...")
print("4) Copia SOLO el valor de code.\n")

code = input("Pega aquí el code: ").strip()

token_response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    },
)

print("\nTOKEN STATUS:", token_response.status_code)
print(token_response.text)

token_response.raise_for_status()

token_data = token_response.json()

TOKEN_OUTPUT.write_text(
    json.dumps(token_data, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nToken guardado en:")
print(TOKEN_OUTPUT)
print("\nScopes concedidos:")
print(token_data.get("scope"))