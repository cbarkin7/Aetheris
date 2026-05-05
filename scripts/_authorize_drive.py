"""
Intercambia el authorization code por tokens con scope /auth/drive completo.
Uso (tras abrir la URL de _check_drive_scope.py en el navegador):

  python scripts/_authorize_drive.py <code>

El codigo aparece en la URL de redireccion como:
  http://localhost/?code=XXXX&scope=...
Copia solo la parte XXXX (desde 'code=' hasta '&scope' o fin de URL).
"""
import json
import sys
import time
import requests
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
SECRET_FILE   = ROOT / "data" / "google" / "client_secret_aetheris.json"
DRIVE_TOKEN   = ROOT / "data" / "google" / ".drive-token.json"
ENV_FILE      = ROOT / ".env"

if len(sys.argv) < 2:
    print("Uso: python scripts/_authorize_drive.py <code>")
    print()
    print("Primero ejecuta: python scripts/_check_drive_scope.py")
    print("Abre la URL indicada, aprueba los permisos y copia el codigo de la URL de redireccion.")
    sys.exit(1)

code = sys.argv[1].strip()
if code.startswith("http"):
    # Si el usuario pego la URL completa, extraer el code
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(code)
    params = parse_qs(parsed.query)
    code = params.get("code", [""])[0]
    if not code:
        print("ERROR: No se pudo extraer el code de la URL. Copia solo el valor del parametro code=XXXX")
        sys.exit(1)
    print(f"Codigo extraido de la URL: {code[:20]}...")

print(f"Intercambiando codigo: {code[:20]}...")

creds_data    = json.loads(SECRET_FILE.read_text())
creds         = creds_data.get("installed") or creds_data.get("web")
CLIENT_ID     = creds["client_id"]
CLIENT_SECRET = creds["client_secret"]

resp = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code":          code,
        "grant_type":    "authorization_code",
        "redirect_uri":  "http://localhost",
    },
    timeout=15,
)
print(f"Status: {resp.status_code}")
body = resp.json()

if not resp.ok:
    print(f"ERROR: {body}")
    sys.exit(1)

access_token  = body["access_token"]
refresh_token = body.get("refresh_token")
expires_in    = body.get("expires_in", 3599)
expiry_ms     = int((time.time() + expires_in) * 1000)

if not refresh_token:
    print("AVISO: Google no devolvio un nuevo refresh_token.")
    print("Esto puede ocurrir si el token ya existe y no se uso prompt=consent.")
    print("Intentando conservar el refresh_token actual del .drive-token.json...")
    try:
        existing = json.loads(DRIVE_TOKEN.read_text())
        refresh_token = existing.get("refresh_token")
    except Exception:
        pass
    if not refresh_token:
        print("ERROR: No hay refresh_token disponible. Ejecuta _check_drive_scope.py de nuevo.")
        sys.exit(1)

# Escribir .drive-token.json actualizado
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
print(f"\n.drive-token.json actualizado con nuevo refresh_token y access_token.")
print(f"expiry_date: {expiry_ms}  (~{expires_in}s)")

# Actualizar .env con el nuevo refresh_token
if refresh_token:
    env_text = ENV_FILE.read_text(encoding="utf-8")
    new_lines = []
    updated = False
    for line in env_text.splitlines():
        if line.startswith("GOOGLE_REFRESH_TOKEN="):
            new_lines.append(f'GOOGLE_REFRESH_TOKEN="{refresh_token}"')
            updated = True
        else:
            new_lines.append(line)
    if updated:
        ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"GOOGLE_REFRESH_TOKEN actualizado en .env")
    else:
        print(f"AVISO: GOOGLE_REFRESH_TOKEN no encontrado en .env — actualiza manualmente:")
        print(f'  GOOGLE_REFRESH_TOKEN="{refresh_token}"')

# Verificar scopes del nuevo token
info = requests.get(
    f"https://oauth2.googleapis.com/tokeninfo?access_token={access_token}",
    timeout=10,
).json()
scope_str  = info.get("scope", "")
scope_list = scope_str.split()

print("\nScopes del nuevo token:")
for s in scope_list:
    print(f"  {s}")

has_drive_full = any(s.endswith("/auth/drive") for s in scope_list)
print()
print("Drive (full):", "OK - listo para crear, editar y eliminar archivos." if has_drive_full else "FALTA - solo lectura disponible.")

if not has_drive_full:
    print()
    print("El scope /auth/drive no fue concedido. Asegurate de haber aprobado todos los permisos")
    print("cuando se abrio la pantalla de autorizacion de Google.")
