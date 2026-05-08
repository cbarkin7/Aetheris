#!/bin/bash
# =============================================================================
# AETHERIS — Entrypoint para contenedor Docker / Hugging Face Spaces
#
# Pasos:
#   1. Decodificar GOOGLE_CLIENT_SECRET_JSON (base64) → fichero JSON en data/google/
#   2. Escribir .gmail-token.json y .calendar-token.json con el REFRESH_TOKEN
#   3. Crear directorios de datos necesarios
#   4. Iniciar supervisord (FastAPI + Streamlit)
# =============================================================================

set -e

echo "[entrypoint] Iniciando AETHERIS..."

# ---------------------------------------------------------------------------
# 1. Credenciales de Google
# ---------------------------------------------------------------------------

GOOGLE_DIR="${GOOGLE_CREDENTIALS_DIR:-/app/data/google}"
mkdir -p "$GOOGLE_DIR"

# Decodificar el contenido del client_secret JSON (variable base64 opcional)
# Uso en HF Spaces: ajustar secreto GOOGLE_CLIENT_SECRET_JSON con el contenido
# base64 del fichero client_secret_aetheris.json
if [ -n "$GOOGLE_CLIENT_SECRET_JSON" ]; then
    echo "[entrypoint] Decodificando GOOGLE_CLIENT_SECRET_JSON..."
    echo "$GOOGLE_CLIENT_SECRET_JSON" | base64 -d > "$GOOGLE_DIR/client_secret_aetheris.json"
    echo "[entrypoint] client_secret_aetheris.json escrito en $GOOGLE_DIR"
else
    echo "[entrypoint] GOOGLE_CLIENT_SECRET_JSON no definido — se asume fichero preexistente"
fi

# Escribir .calendar-token.json si se han proporcionado credenciales OAuth
if [ -n "$GOOGLE_REFRESH_TOKEN" ] && [ -f "$GOOGLE_DIR/client_secret_aetheris.json" ]; then
    echo "[entrypoint] Escribiendo tokens Google OAuth..."

    # Leer client_id y client_secret del fichero (via python inline)
    CLIENT_DATA=$(python3 -c "
import json, sys
try:
    with open('$GOOGLE_DIR/client_secret_aetheris.json') as f:
        d = json.load(f)
    inner = d.get('installed') or d.get('web') or {}
    print(inner.get('client_id',''), inner.get('client_secret',''))
except Exception as e:
    print('', '')
")
    CLIENT_ID=$(echo "$CLIENT_DATA" | awk '{print $1}')
    CLIENT_SECRET=$(echo "$CLIENT_DATA" | awk '{print $2}')

    # .calendar-token.json
    CALENDAR_TOKEN="$GOOGLE_DIR/.calendar-token.json"
    if [ ! -f "$CALENDAR_TOKEN" ]; then
        python3 -c "
import json
token = {
    'token': None,
    'refresh_token': '$GOOGLE_REFRESH_TOKEN',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'client_id': '$CLIENT_ID',
    'client_secret': '$CLIENT_SECRET',
    'scopes': ['https://www.googleapis.com/auth/calendar'],
    'expiry': None
}
with open('$CALENDAR_TOKEN', 'w') as f:
    json.dump(token, f, indent=2)
print('[entrypoint] .calendar-token.json creado')
"
    else
        echo "[entrypoint] .calendar-token.json ya existe — reutilizando"
    fi

    # .gmail-token.json
    GMAIL_TOKEN="$GOOGLE_DIR/.gmail-token.json"
    if [ ! -f "$GMAIL_TOKEN" ]; then
        python3 -c "
import json
token = {
    'access_token': None,
    'refresh_token': '$GOOGLE_REFRESH_TOKEN',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'client_id': '$CLIENT_ID',
    'client_secret': '$CLIENT_SECRET',
    'scopes': [
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send'
    ],
    'expiry': None
}
with open('$GMAIL_TOKEN', 'w') as f:
    json.dump(token, f, indent=2)
print('[entrypoint] .gmail-token.json creado')
"
    else
        echo "[entrypoint] .gmail-token.json ya existe — reutilizando"
    fi

    # .drive-token.json (mismo refresh token, scope Drive)
    DRIVE_TOKEN="$GOOGLE_DIR/.drive-token.json"
    if [ ! -f "$DRIVE_TOKEN" ]; then
        python3 -c "
import json
token = {
    'token': None,
    'refresh_token': '$GOOGLE_REFRESH_TOKEN',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'client_id': '$CLIENT_ID',
    'client_secret': '$CLIENT_SECRET',
    'scopes': ['https://www.googleapis.com/auth/drive'],
    'expiry': None
}
with open('$DRIVE_TOKEN', 'w') as f:
    json.dump(token, f, indent=2)
print('[entrypoint] .drive-token.json creado')
"
    else
        echo "[entrypoint] .drive-token.json ya existe — reutilizando"
    fi
else
    echo "[entrypoint] GOOGLE_REFRESH_TOKEN no definido — herramientas Google desactivadas"
fi

# ---------------------------------------------------------------------------
# 2. Directorios de datos
# ---------------------------------------------------------------------------
mkdir -p "${CHROMA_PERSIST_DIR:-/app/data/chroma_db}"
mkdir -p "$(dirname "${SQLITE_CHECKPOINTS_PATH:-/app/data/sqlite/checkpoints.db}")"
mkdir -p "${UPLOADS_DIR:-/app/data/uploads}"

echo "[entrypoint] Directorios de datos listos"

# ---------------------------------------------------------------------------
# 3. Iniciar supervisord
# ---------------------------------------------------------------------------
echo "[entrypoint] Iniciando supervisord (FastAPI :8000 + Streamlit :7860)..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/aetheris.conf
