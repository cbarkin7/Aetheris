"""
AETHERIS — Configuración de la aplicación.
Ajustes tipados cargados desde variables de entorno / fichero .env.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Proveedor LLM principal: OpenAI
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="Clave API de OpenAI")

    # -------------------------------------------------------------------------
    # AWS Bedrock (fallback automático si OpenAI falla)
    # -------------------------------------------------------------------------
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID para Bedrock")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key para Bedrock")
    aws_session_token: str = Field(default="", description="AWS Session Token (credenciales temporales)")
    aws_region: str = Field(default="eu-west-1", description="Región AWS para Bedrock")
    bedrock_model_id: str = Field(
        default="anthropic.claude-sonnet-4-5-20250929-v1:0",
        description="ID del modelo Bedrock para fallback",
    )

    # -------------------------------------------------------------------------
    # HuggingFace Hub (descarga de modelos)
    # -------------------------------------------------------------------------
    hf_token: str = Field(default="", description="Token HuggingFace (Read) para evitar rate-limiting al descargar modelos")

    # -------------------------------------------------------------------------
    # Speech-to-Text local (faster-whisper, sin coste de API)
    # Tamaños disponibles: tiny, base, small, medium, large-v2, large-v3
    # -------------------------------------------------------------------------
    whisper_model_size: str = Field(
        default="small",
        description="Tamaño del modelo faster-whisper: tiny, base, small, medium, large-v2",
    )
    whisper_device: str = Field(
        default="cpu",
        description="Dispositivo para inferencia: cpu o cuda",
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="Tipo de cómputo: int8 (CPU rápido), float16 (GPU), float32",
    )

    # -------------------------------------------------------------------------
    # LangSmith
    # -------------------------------------------------------------------------
    langchain_tracing_v2: bool = Field(default=True)
    langsmith_api_key: str = Field(default="", description="Clave API de LangSmith")
    langsmith_project: str = Field(default="aetheris")
    langsmith_endpoint: str = Field(
        default="https://eu.api.smith.langchain.com",
        description="Endpoint de LangSmith (usar https://eu.api.smith.langchain.com para la región EU)",
    )

    # -------------------------------------------------------------------------
    # Servicios externos (MCP)
    # -------------------------------------------------------------------------
    tavily_api_key: str = Field(default="", description="Clave API de Tavily Search")

    # Google OAuth2
    # google_client_secret_file apunta al JSON descargado de Google Cloud Console (tipo Desktop app).
    # Si el fichero existe, se usa directamente como GOOGLE_OAUTH_CREDENTIALS para los servidores MCP.
    # client_id y client_secret se leen del fichero; los campos individuales son opcionales.
    google_client_secret_file: str = Field(
        default="data/google/client_secret_aetheris.json",
        description="Ruta al fichero client_secret_*.json descargado de Google Cloud Console",
    )
    google_client_id: str = Field(default="", description="OAuth2 Client ID (fallback si no hay fichero)")
    google_client_secret: str = Field(default="", description="OAuth2 Client Secret (fallback)")
    google_refresh_token: str = Field(default="", description="Refresh token obtenido tras el flujo OAuth")
    google_credentials_dir: str = Field(
        default="data/google",
        description="Directorio donde se escriben los ficheros de token authorized_user",
    )
    gmail_mcp_url: str = Field(
        default="http://localhost:30000/mcp",
        description="URL del servidor Gmail MCP HTTP (Bearer auth)",
    )

    # -------------------------------------------------------------------------
    # Mem0 (memoria conversacional)
    # -------------------------------------------------------------------------
    mem0_api_key: str = Field(default="", description="Clave API de mem0.ai (vacío = modo local)")
    mem0_org_id: str = Field(default="", description="ID de organización de mem0 (opcional)")
    mem0_project_id: str = Field(default="", description="ID de proyecto de mem0 (opcional)")

    # -------------------------------------------------------------------------
    # Aplicación
    # -------------------------------------------------------------------------
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="change-me-in-production")

    # -------------------------------------------------------------------------
    # Rutas de datos
    # -------------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="data/chroma_db")
    sqlite_checkpoints_path: str = Field(default="data/sqlite/checkpoints.db")
    sqlite_memory_path: str = Field(default="data/sqlite/memory.db")
    uploads_dir: str = Field(default="data/uploads")

    # -------------------------------------------------------------------------
    # Ajustes del modelo
    # -------------------------------------------------------------------------
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    llm_temperature: float = Field(default=0.0)

    # -------------------------------------------------------------------------
    # Ajustes RAG
    # -------------------------------------------------------------------------
    rag_chunk_size: int = Field(default=1000)
    rag_chunk_overlap: int = Field(default=200)
    rag_retrieval_k: int = Field(default=5)
    rag_score_threshold: float = Field(default=0.3)

    # -------------------------------------------------------------------------
    # Guardrails
    # -------------------------------------------------------------------------
    guardrails_enabled: bool = Field(default=True)
    guardrails_max_input_length: int = Field(default=8000)
    guardrails_redact_pii: bool = Field(default=True)
    guardrails_block_injections: bool = Field(default=True)

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:8501")

    # -------------------------------------------------------------------------
    # Streamlit
    # -------------------------------------------------------------------------
    streamlit_port: int = Field(default=8501)
    api_base_url: str = Field(default="http://localhost:8000")

    # -------------------------------------------------------------------------
    # Propiedades calculadas
    # -------------------------------------------------------------------------
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def bedrock_available(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def mem0_cloud_mode(self) -> bool:
        return bool(self.mem0_api_key)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level debe ser uno de {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia singleton de Settings (cacheada tras la primera llamada)."""
    return Settings()
