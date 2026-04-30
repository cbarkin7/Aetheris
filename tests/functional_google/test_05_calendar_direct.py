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

token_response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    },
)

print("TOKEN STATUS:", token_response.status_code)
print(token_response.text)

token_response.raise_for_status()

access_token = token_response.json()["access_token"]

calendar_response = requests.get(
    "https://www.googleapis.com/calendar/v3/users/me/calendarList",
    headers={"Authorization": f"Bearer {access_token}"},
)

print("CALENDAR STATUS:", calendar_response.status_code)
print(calendar_response.text[:2000])