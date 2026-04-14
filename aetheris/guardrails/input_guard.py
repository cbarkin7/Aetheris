"""
Guardrail de entrada — valida y sanea los mensajes del usuario antes del procesamiento.

Comprobaciones (bilingüe EN + ES):
  1. Límite de longitud de entrada
  2. Detección de inyección de prompts (patrones en inglés y español)
  3. Redacción de PII (email, teléfono, DNI/NIE, tarjeta de crédito, IBAN)
"""
import logging
import re

from aetheris.guardrails.schemas import GuardResult, redact_pii

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patrones de inyección de prompts (EN + ES) — compilados a nivel de módulo
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions_en", re.compile(
        r"(?i)(ignore|disregard|forget|override|bypass)\s+.{0,30}"
        r"(instruction|prompt|system|rule|constraint|guideline)",
    )),
    ("ignore_instructions_es", re.compile(
        r"(?i)(ignora|olvida|descarta|salta|anula|sobrescribe)\s+.{0,30}"
        r"(instrucci[oó]n|prompt|sistema|regla|restricci[oó]n|directriz|indicaci[oó]n)",
    )),
    ("reveal_system_en", re.compile(
        r"(?i)(reveal|show|print|output|display|repeat)\s+.{0,30}"
        r"(system\s*message|system\s*prompt|instruction|hidden|internal)",
    )),
    ("reveal_system_es", re.compile(
        r"(?i)(mu[eé]stra(?:me)?|revela|imprime|ense[ñn]a|repite|dame)\s+.{0,30}"
        r"(mensaje\s*del?\s*sistema|prompt\s*del?\s*sistema|instrucci[oó]n|oculto|interno)",
    )),
    ("role_hijack_en", re.compile(
        r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+role|"
        r"switch\s+to|roleplay\s+as)",
    )),
    ("role_hijack_es", re.compile(
        r"(?i)(ahora\s+eres|act[uú]a\s+como|finge\s+ser|nuevo\s+rol|"
        r"cambia\s+a|simula\s+ser|hazte\s+pasar)",
    )),
    ("code_injection", re.compile(
        r"(?i)(```|exec\(|eval\(|import\s+os|subprocess|__import__|"
        r"os\.system|rm\s+-rf|import\s+shutil)",
    )),
]

# ---------------------------------------------------------------------------
# Patrones de PII — compilados a nivel de módulo
# ---------------------------------------------------------------------------
_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[\s\-]?(?:\d{4}[\s\-]?){4,7}\d{0,4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b"),
    "dni_nie_es": re.compile(r"\b[XYZxyz]?\d{7,8}[A-Za-z]\b"),
    "phone_intl": re.compile(
        r"(?<!\d)(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)\d{3,4}[\s\-]?\d{3,4}(?!\d)"
    ),
}

_PII_REPLACEMENTS: dict[str, str] = {
    "email": "[EMAIL_REDACTADO]",
    "phone_intl": "[TELEFONO_REDACTADO]",
    "ssn": "[SSN_REDACTADO]",
    "dni_nie_es": "[DNI_REDACTADO]",
    "credit_card": "[TARJETA_REDACTADA]",
    "iban": "[IBAN_REDACTADO]",
}

MAX_INPUT_LENGTH = 8000


class InputGuard:
    """
    Guardrail de entrada sin estado.

    Uso:
        guard = InputGuard()
        result = guard.check("mensaje del usuario")
        if result.blocked:
            return f"Bloqueado: {result.violations}"
        safe_text = result.sanitized_text
    """

    def __init__(
        self,
        max_length: int = MAX_INPUT_LENGTH,
        redact_pii: bool = True,
        block_injections: bool = True,
    ):
        self.max_length = max_length
        self._redact_pii = redact_pii
        self.block_injections = block_injections

    def check(self, text: str) -> GuardResult:
        # 1. Comprobación de longitud
        if len(text) > self.max_length:
            return GuardResult(
                passed=False,
                sanitized_text=text[: self.max_length],
                violations=[f"input_too_long:{len(text)}>{self.max_length}"],
            )

        # 2. Detección de inyección de prompts
        violations: list[str] = []
        if self.block_injections:
            for name, pattern in _INJECTION_PATTERNS:
                if pattern.search(text):
                    violations.append(f"prompt_injection:{name}")
                    logger.warning("Inyección de prompt detectada [%s]: '%.80s…'", name, text)

        if violations:
            return GuardResult(passed=False, sanitized_text=text, violations=violations)

        # 3. Redacción de PII (no bloquea — redacta y continúa)
        sanitized = text
        redactions: dict[str, int] = {}
        if self._redact_pii:
            sanitized, redactions = redact_pii(text, _PII_PATTERNS, _PII_REPLACEMENTS)
            for pii_type, count in redactions.items():
                logger.info("Redactado %d coincidencia(s) de %s en la entrada", count, pii_type)

        return GuardResult(passed=True, sanitized_text=sanitized, violations=[], redactions=redactions)
