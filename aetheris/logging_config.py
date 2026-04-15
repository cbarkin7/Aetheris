"""
AETHERIS — Configuración centralizada de logging.

Formato colorizado por nivel con columnas fijas para facilitar la lectura
de trazas en terminal durante la ejecución del agente.

Uso:
    from aetheris.logging_config import setup_logging
    setup_logging(level="INFO")  # llamar una sola vez al arrancar
"""
import logging
import sys

# ---------------------------------------------------------------------------
# Códigos ANSI
# ---------------------------------------------------------------------------
_R     = "\033[0m"          # reset
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREY  = "\033[38;5;245m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_YELL  = "\033[33m"
_RED   = "\033[31m"
_MAG   = "\033[35m"
_BLUE  = "\033[34m"
_WHITE = "\033[97m"

_LEVEL_COLORS = {
    "DEBUG":    _GREY,
    "INFO":     _GREEN,
    "WARNING":  _YELL,
    "ERROR":    _RED,
    "CRITICAL": _MAG,
}

# Prefijos de sección → color de etiqueta
_SECTION_COLORS = {
    "[GUARDRAIL": _MAG,
    "[MANAGER":   _BLUE,
    "[PLAN]":     _BLUE,
    "[RAG]":      _CYAN,
    "[WEB":       _CYAN,
    "[HITL]":     _YELL,
    "[GOOGLE]":   _YELL,
    "[LLM]":      _GREEN,
    "[MEMORIA":   _CYAN,
    "[MCP]":      _CYAN,
    "[CHECKP":    _GREY,
    "[API]":      _WHITE,
    "[SISTEMA]":  _WHITE,
}


def _color_section(msg: str) -> str:
    """Aplica color ANSI a la etiqueta de sección al inicio del mensaje."""
    for prefix, color in _SECTION_COLORS.items():
        if msg.startswith(prefix):
            # Colorear solo la primera palabra/etiqueta hasta el espacio o │
            end = msg.find(" ")
            if end == -1:
                return f"{color}{_BOLD}{msg}{_R}"
            return f"{color}{_BOLD}{msg[:end]}{_R}{msg[end:]}"
    return msg


class AetherisFormatter(logging.Formatter):
    """
    Formatter con columnas fijas y colores ANSI:

    HH:MM:SS │ LEVEL    │ módulo                        │ [SECCIÓN] → función | estado | detalles
    """

    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color   = _LEVEL_COLORS.get(record.levelname, _R)
        time_s  = self.formatTime(record, self.DATE_FMT)
        level_s = f"{color}{_BOLD}{record.levelname:<8}{_R}"

        # Acortar nombre del módulo: aetheris.agent.nodes → agent.nodes
        module  = record.name.replace("aetheris.", "")
        mod_s   = f"{_DIM}{module:<28}{_R}"

        msg = record.getMessage()
        msg = _color_section(msg)

        line = f"{_GREY}{time_s}{_R} │ {level_s} │ {mod_s} │ {msg}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO") -> None:
    """
    Configura el sistema de logging de AETHERIS.

    Debe llamarse UNA sola vez al arrancar la aplicación (lifespan de FastAPI
    o al inicio del script). Uvicorn vuelve a llamarla en cada recarga caliente,
    por lo que se eliminan handlers previos antes de añadir el nuevo.

    Args:
        level: Nivel mínimo para los logs de AETHERIS ("DEBUG" | "INFO" | "WARNING").
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)      # el handler filtra por nivel

    # Eliminar handlers previos (recarga uvicorn --reload)
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(AetherisFormatter())
    root.addHandler(handler)

    # ── Silenciar librerías ruidosas ─────────────────────────────────────────
    for noisy in (
        "httpx", "httpcore", "openai._base_client",
        "chromadb", "chromadb.telemetry", "chromadb.segment",
        "urllib3", "urllib3.connectionpool",
        "langchain_core", "langchain_text_splitters",
        "langgraph", "uvicorn.access",
        "multipart", "botocore", "boto3",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Mantener WARNING para openai (errores de cuota, rate-limit) y uvicorn
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    # Nuestro paquete en nivel configurable
    logging.getLogger("aetheris").setLevel(
        getattr(logging, level.upper(), logging.INFO)
    )

    logging.getLogger(__name__).info(
        "[SISTEMA] → setup_logging | completado | nivel=%s", level
    )
