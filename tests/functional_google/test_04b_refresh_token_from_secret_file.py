from dotenv import load_dotenv
from pathlib import Path
import os
import json
import requests

load_dotenv()

secret_file = os.getenv("GOOGLE_CLIENT_SECRET_FILE")
refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")

data = json.loads(Path(secret_file).read_text(encoding="utf-8"))
client_data = data.get("installed") or data.get("web")

client_id = client_data["client_id"]
client_secret = client_data["client_secret"]

response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    },
)

print("Status:", response.status_code)
print(response.text)