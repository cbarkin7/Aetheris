from dotenv import load_dotenv
from pathlib import Path
import os
import json
import requests

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
secret_file = ROOT / "data" / "google" / "client_secret_aetheris.json"

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
)

print("STATUS:", response.status_code)
print(response.text)