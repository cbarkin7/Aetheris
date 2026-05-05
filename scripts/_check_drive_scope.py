"""
Verifica los scopes del refresh_token y regenera .drive-token.json.
Ejecutar desde la raiz del proyecto:
  python scripts/_check_drive_scope.py
"""
import json, sys, time, requests, urllib.parse
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
SECRET_FILE   = ROOT / "data" / "google" / "client_secret_aetheris.json"
DRIVE_TOKEN   = ROOT / "data" / "google" / ".drive-token.json"
ENV_FILE      = ROOT / ".env"

creds_data    = json.loads(SECRET_FILE.read_text())
creds         = creds_data.get("installed") or creds_data.get("web")
CLIENT_ID     = creds["client_id"]
CLIENT_SECRET = creds["client_secret"]

refresh_token = None
for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    if line.startswith("GOOGLE_REFRESH_TOKEN="):
        refresh_token = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

if not refresh_token:
    print("ERROR: GOOGLE_REFRESH_TOKEN no encontrado en .env")
    sys.exit(1)

print(f"refresh_token: {refresh_token[:20]}...")

# Refresco sin scope (Google usa los scopes originales del token)
resp = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type":    "refresh_token",
    },
    timeout=15,
)
print(f"Status refresco: {resp.status_code}")
body = resp.json()

if not resp.ok:
    print(f"Error: {body}")
    sys.exit(1)

access_token = body["access_token"]
expires_in   = body.get("expires_in", 3599)
expiry_ms    = int((time.time() + expires_in) * 1000)

# Inspeccionar scopes
info = requests.get(
    f"https://oauth2.googleapis.com/tokeninfo?access_token={access_token}",
    timeout=10,
).json()
scope_str    = info.get("scope", "")
scope_list   = scope_str.split()

print("\nScopes del refresh_token:")
for s in scope_list:
    print(f"  {s}")

has_calendar      = any("calendar"       in s for s in scope_list)
has_drive_full    = any(s.endswith("/auth/drive") for s in scope_list)
has_drive_file    = any("drive.file"     in s for s in scope_list)
has_drive_read    = any("drive.readonly" in s for s in scope_list)
has_gmail         = any("gmail"          in s for s in scope_list)

print(f"\n  calendar       : {'OK' if has_calendar   else 'FALTA'}")
print(f"  drive (full)   : {'OK' if has_drive_full  else 'FALTA'}")
print(f"  drive.file     : {'OK' if has_drive_file  else 'FALTA'}")
print(f"  drive.readonly : {'OK' if has_drive_read  else 'FALTA'}")
print(f"  gmail          : {'OK' if has_gmail       else 'FALTA'}")

# Siempre escribir el token con access_token activo para evitar el flujo OAuth
token = {
    "type":          "authorized_user",
    "client_id":     CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "refresh_token": refresh_token,
    "access_token":  access_token,
    "expiry_date":   expiry_ms,
    "token_uri":     "https://oauth2.googleapis.com/token",
}
DRIVE_TOKEN.write_text(json.dumps(token, indent=2), encoding="utf-8")
print(f"\n.drive-token.json actualizado con access_token activo.")
print(f"expiry_date: {expiry_ms}  (~{expires_in}s)")

if not has_drive_full:
    SCOPE = " ".join([
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/gmail.modify",
    ])
    url = (
        "https://accounts.google.com/o/oauth2/auth"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri=http://localhost"
        f"&response_type=code"
        f"&scope={urllib.parse.quote(SCOPE)}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    print()
    print("AVISO: El token tiene drive.file + drive.readonly pero NO /auth/drive completo.")
    print("Las lecturas y listados funcionaran. Subida/eliminacion/edicion puede fallar.")
    print()
    print("Para re-autorizar con Drive completo, abre esta URL en el navegador:")
    print()
    print(url)
    print()
    print("Tras aprobar, Google redirige a http://localhost/?code=XXXX")
    print("Ejecuta: python scripts/_authorize_drive.py <code>")
else:
    print("\nDrive (full) OK - no es necesario re-autorizar.")
