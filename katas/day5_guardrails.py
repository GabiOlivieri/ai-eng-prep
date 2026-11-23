"""Día 5 (40 min): redacción reversible de PII y detección simple de injection."""
from dataclasses import dataclass, field


@dataclass
class Redaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # "<EMAIL_1>" -> valor original

    def restore(self, text: str) -> str:
        # TODO: reemplazar placeholders por los valores originales.
        raise NotImplementedError


def luhn_valid(number: str) -> bool:
    raise NotImplementedError


def redact_pii(text: str) -> Redaction:
    # TODO: emails, teléfonos, CUIT/DNI y tarjetas (solo si pasan Luhn) -> <EMAIL_1>, <CARD_1>, ...
    raise NotImplementedError


def looks_like_injection(text: str) -> bool:
    # TODO: heurísticas (ignorá instrucciones, revelá el system prompt, URLs de exfiltración...).
    # Después sumá un clasificador con un modelo chico y structured output.
    raise NotImplementedError
