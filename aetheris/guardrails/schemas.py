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
    # pii_map reversible: {placeholder: valor_original}
    # Permite a google_action_node restaurar los datos reales antes de invocar tools.
    redactions: dict[str, str] = Field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.passed


def redact_pii(
    text: str,
    patterns: dict[str, re.Pattern],
    replacements: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """
    Aplica redacción de PII al texto construyendo un mapa reversible.

    Usa placeholders numerados para múltiples ocurrencias del mismo tipo:
      1ª ocurrencia → [EMAIL_REDACTADO]
      2ª ocurrencia → [EMAIL_REDACTADO_2]
      ...
    Si el mismo valor aparece varias veces, reutiliza el mismo placeholder.

    Args:
        text: Texto a sanear.
        patterns: Diccionario {nombre: patrón compilado}.
        replacements: Diccionario {nombre: placeholder base, e.g. "[EMAIL_REDACTADO]"}.

    Returns:
        (texto_saneado, pii_map) donde pii_map == {placeholder: valor_original}.
        El mapa permite a google_action_node restaurar los datos reales antes de
        invocar herramientas (Gmail, Calendar, Drive) que necesitan valores válidos.
    """
    pii_map: dict[str, str] = {}
    sanitized = text

    for pii_type, pattern in patterns.items():
        base = replacements[pii_type]   # p.ej. "[EMAIL_REDACTADO]"
        base_prefix = base.rstrip("]") # "[EMAIL_REDACTADO"

        def _replace(m: re.Match, _base: str = base, _prefix: str = base_prefix) -> str:
            original = m.group(0)

            # Reutilizar placeholder si el mismo valor ya fue redactado
            for ph, val in pii_map.items():
                if val == original:
                    return ph

            # Contar cuántos placeholders de este tipo ya existen para numerar
            existing_count = sum(
                1 for ph in pii_map
                if ph == _base or ph.startswith(_prefix + "_")
            )
            placeholder = _base if existing_count == 0 else f"{_prefix}_{existing_count + 1}]"
            pii_map[placeholder] = original
            return placeholder

        sanitized = pattern.sub(_replace, sanitized)

    return sanitized, pii_map
