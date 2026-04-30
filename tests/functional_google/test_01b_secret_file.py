from dotenv import load_dotenv
from pathlib import Path
import os
import json

load_dotenv()

secret_file = os.getenv("GOOGLE_CLIENT_SECRET_FILE")

print("GOOGLE_CLIENT_SECRET_FILE:", secret_file)

path = Path(secret_file)
print("Existe:", path.exists())
print("Ruta absoluta:", path.resolve())

data = json.loads(path.read_text(encoding="utf-8"))

print("Claves principales:", data.keys())

client_data = data.get("installed") or data.get("web")

print("Tipo:", "installed" if "installed" in data else "web")
print("CLIENT_ID existe:", bool(client_data.get("client_id")))
print("CLIENT_SECRET existe:", bool(client_data.get("client_secret")))
print("Redirect URIs:", client_data.get("redirect_uris"))