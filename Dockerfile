# =============================================================================
# AETHERIS — Dockerfile para Hugging Face Spaces
#
# Arquitectura:
#   - FastAPI (uvicorn) en el puerto interno 8000
#   - Streamlit en el puerto publico 7860  (HF Spaces expone solo este)
#   - supervisord gestiona ambos procesos
#   - Usuario no-root uid=1000 (requisito HF Spaces)
#   - Node.js 18 incluido para los servidores MCP npx (Calendar, Drive)
# =============================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# Variables de entorno de construccion
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Dependencias del sistema + Node.js 18
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        ca-certificates \
        supervisor \
        ffmpeg \
        # para compilar algunas dependencias Python si es necesario
        build-essential \
    && \
    # Node.js 18 via NodeSource
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    # Limpieza
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Usuario no-root (uid 1000 requerido por HF Spaces)
# ---------------------------------------------------------------------------
RUN useradd -m -u 1000 aetheris

# ---------------------------------------------------------------------------
# Directorio de trabajo
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Instalar dependencias Python (como root para instalar en site-packages)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Copiar codigo fuente
# ---------------------------------------------------------------------------
COPY --chown=aetheris:aetheris . .

# ---------------------------------------------------------------------------
# Directorios de datos persistibles (volumenes en HF Spaces = /data)
# En HF Spaces la carpeta /data es persistente; crear symlinks si fuera
# necesario. Por defecto usamos rutas relativas que caen en /app/data.
# ---------------------------------------------------------------------------
RUN mkdir -p \
        /app/data/chroma_db \
        /app/data/sqlite \
        /app/data/uploads \
        /app/data/google \
    && chown -R aetheris:aetheris /app/data

# ---------------------------------------------------------------------------
# Copiar ficheros de configuracion de procesos
# ---------------------------------------------------------------------------
COPY supervisord.conf /etc/supervisor/conf.d/aetheris.conf
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------------------------------------------------------------------------
# Cambiar a usuario no-root
# ---------------------------------------------------------------------------
USER aetheris

# ---------------------------------------------------------------------------
# Puerto expuesto (Streamlit — unico puerto publico en HF Spaces)
# ---------------------------------------------------------------------------
EXPOSE 7860

# ---------------------------------------------------------------------------
# Variables de entorno con valores por defecto para produccion
# ---------------------------------------------------------------------------
ENV APP_ENV=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_PORT=7860 \
    API_BASE_URL=http://localhost:8000 \
    CORS_ORIGINS=http://localhost:7860 \
    CHROMA_PERSIST_DIR=/app/data/chroma_db \
    SQLITE_CHECKPOINTS_PATH=/app/data/sqlite/checkpoints.db \
    SQLITE_MEMORY_PATH=/app/data/sqlite/memory.db \
    UPLOADS_DIR=/app/data/uploads \
    GOOGLE_CLIENT_SECRET_FILE=/app/data/google/client_secret_aetheris.json \
    GOOGLE_CREDENTIALS_DIR=/app/data/google

# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
ENTRYPOINT ["/entrypoint.sh"]
