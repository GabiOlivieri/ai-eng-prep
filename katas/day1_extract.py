"""Día 1, ejercicio 2 (40 min): extracción estructurada con loop de reparación.

Solución de referencia. Ideas clave:
- El schema JSON sale de Pydantic (`model_json_schema()`): una sola fuente de verdad.
- Se valida con `model_validate_json`, que parsea y valida en un paso (JSON roto también
  es un `ValidationError`, no un `json.JSONDecodeError`).
- Si falla, se le devuelve al modelo SU respuesta (rol assistant) y el error (rol user).
  Ver el error concreto ("priority: Input should be <= 5") es lo que le permite corregirse.
- Total de llamadas = 1 + max_repairs. Después: excepción propia, encadenando la causa.
- Muchos modelos envuelven el JSON en ```json ... ``` o, como Qwen3, agregan <think>...</think>.
  `_clean` saca ese ruido antes de validar (barato y evita gastar un intento de reparación).
- En producción, si el proveedor lo soporta, sumá structured output nativo
  (`response_format={"type": "json_schema", ...}`); el loop de reparación sigue siendo la red.
"""
import json
import re
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, ValidationError

# Recibe la lista de mensajes y devuelve el texto crudo del modelo.
LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


class SupportTicket(BaseModel):
    category: Literal["billing", "bug", "feature", "other"]
    priority: int = Field(ge=1, le=5)
    customer_email: EmailStr | None = None
    summary: str = Field(max_length=200)


class ExtractionFailed(Exception):
    pass


SYSTEM_PROMPT = """Sos un sistema que clasifica emails de soporte.
Respondé SOLO con un objeto JSON válido (sin texto extra, sin markdown) que cumpla este JSON Schema:
{schema}
Reglas: priority va de 1 (baja) a 5 (urgente). Si no hay email del cliente, usá null.
summary: una oración corta en el idioma del email."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _clean(raw: str) -> str:
    return _FENCE.sub("", _THINK.sub("", raw).strip()).strip()


async def extract_ticket(email_text: str, llm: LLMCall, max_repairs: int = 2) -> SupportTicket:
    schema = json.dumps(SupportTicket.model_json_schema(), ensure_ascii=False)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": email_text},
    ]
    last_error: ValidationError | None = None

    for _ in range(1 + max_repairs):
        raw = await llm(messages)
        try:
            return SupportTicket.model_validate_json(_clean(raw))
        except ValidationError as err:
            last_error = err
            # Copiamos la conversación: no mutamos listas que el caller pudo haber guardado.
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Tu respuesta no pasó la validación:\n"
                        f"{err}\n"
                        "Devolvé SOLO el JSON corregido."
                    ),
                },
            ]

    raise ExtractionFailed(
        f"No se pudo extraer un ticket válido tras {1 + max_repairs} intentos"
    ) from last_error
