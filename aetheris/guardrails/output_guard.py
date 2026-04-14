"""
Guardrail de salida — valida y sanea las respuestas del asistente antes de entregarlas.

Comprobaciones (bilingüe EN + ES):
  1. Truncado de respuestas excesivamente largas
  2. Redacción de PII filtrada en la salida
  3. Detección de contenido inseguro (claves API, contraseñas, prompts internos)
"""
import logging
import re

from aetheris.guardrails.input_guard import _PII_PATTERNS, _PII_REPLACEMENTS
from aetheris.guardrails.schemas import GuardResult, redact_pii

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 16000

# Patrones de salida insegura (EN + ES) — compilados a nivel de módulo
_UNSAFE_OUTPUT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("leaked_api_key", re.compile(
        r"(?i)(sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36})"
    )),
    ("leaked_password_en", re.compile(
        r"(?i)(password|passwd|pwd)\s*[:=]\s*\S{6,}"
    )),
    ("leaked_password_es", re.compile(
        r"(?i)(contraseña|clave|passwd)\s*[:=]\s*\S{6,}"
    )),
    ("internal_system_leak_en", re.compile(
        r"(?i)(system\s*prompt|internal\s*instruction|you\s+are\s+an?\s+AI\s+assistant\s+that)"
    )),
    ("internal_system_leak_es", re.compile(
        r"(?i)(prompt\s*del?\s*sistema|instrucci[oó]n\s*interna|eres\s+un\s+asistente\s+de\s+IA\s+que)"
    )),
]


class OutputGuard:
    """
    Guardrail de salida sin estado.

    Uso:
        guard = OutputGuard()
        result = guard.check("respuesta del asistente")
        safe_response = result.sanitized_text
    """

    def __init__(
        self,
        max_length: int = MAX_OUTPUT_LENGTH,
        redact_pii: bool = True,
        block_unsafe: bool = True,
    ):
        self.max_length = max_length
        self._redact_pii = redact_pii
        self.block_unsafe = block_unsafe

    def check(self, text: str) -> GuardResult:
        violations: list[str] = []
        sanitized = text

        # 1. Truncado de respuestas muy largas
        if len(text) > self.max_length:
            sanitized = text[: self.max_length] + "\n\n[Respuesta truncada]"
            violations.append("output_truncated")

        # 2. Detección y redacción de contenido inseguro (antes de PII para
        #    evitar que el patrón de teléfono fragmente claves API/contraseñas)
        if self.block_unsafe:
            for name, pattern in _UNSAFE_OUTPUT_PATTERNS:
                if pattern.search(sanitized):
                    violations.append(f"unsafe_output:{name}")
                    sanitized = pattern.sub(f"[{name.upper()}_REDACTADO]", sanitized)
                    logger.warning("Contenido inseguro redactado en la salida: %s", name)

        # 3. Redacción de PII en la salida
        if self._redact_pii:
            sanitized, redactions = redact_pii(sanitized, _PII_PATTERNS, _PII_REPLACEMENTS)
            for pii_type, count in redactions.items():
                logger.warning("Fuga de PII prevenida: %d %s en la salida", count, pii_type)
        else:
            redactions = {}

        return GuardResult(
            passed=True,  # La salida siempre se entrega (saneada), nunca se bloquea
            sanitized_text=sanitized,
            violations=violations,
            redactions=redactions,
        )
