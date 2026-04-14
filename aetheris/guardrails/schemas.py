"""
Esquemas compartidos y utilidades para los guardrails de AETHERIS.
"""
import re

from pydantic import BaseModel, Field


class GuardResult(BaseModel):
    """Resultado de una comprobación de guardrail."""
    passed: bool = True
    sanitized_text: str = ""
    violations: list[str] = Field(default_factory=list)
    redactions: dict[str, int] = Field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed


def redact_pii(
    text: str,
    patterns: dict[str, re.Pattern],
    replacements: dict[str, str],
) -> tuple[str, dict[str, int]]:
    """
    Aplica redacción de PII al texto.

    Args:
        text: Texto a sanear.
        patterns: Diccionario {nombre: patrón compilado}.
        replacements: Diccionario {nombre: texto de reemplazo}.

    Returns:
        (texto_saneado, conteos_por_tipo)
    """
    redaction_counts: dict[str, int] = {}
    sanitized = text
    for pii_type, pattern in patterns.items():
        matches = pattern.findall(sanitized)
        if matches:
            sanitized = pattern.sub(replacements[pii_type], sanitized)
            redaction_counts[pii_type] = len(matches)
    return sanitized, redaction_counts
