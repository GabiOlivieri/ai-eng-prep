"""Día 1, ejercicio 2 (40 min): extracción estructurada con loop de reparación."""
from collections.abc import Awaitable, Callable
import logging
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, ValidationError
import json

# Recibe la lista de mensajes y devuelve el texto crudo del modelo.
LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


class SupportTicket(BaseModel):
    category: Literal["billing", "bug", "feature", "other"]
    priority: int = Field(ge=1, le=5)
    customer_email: EmailStr | None = None
    summary: str = Field(max_length=200)


class ExtractionFailed(Exception):
    pass


async def extract_ticket(email_text: str, llm: LLMCall, max_repairs: int = 2) -> SupportTicket:
    # TODO: system prompt con el schema (SupportTicket.model_json_schema()),
    # validar con model_validate_json, y si falla reenviar el error al modelo.
    # Después de max_repairs intentos fallidos -> ExtractionFailed.
    messages = [
        {"role": "system", "content": json.dumps(SupportTicket.model_json_schema())},
        {"role": "user", "content": email_text}
    ]
    last_error = None
    response = None

    for _ in range(1 + max_repairs):
        try:
             response_text = await llm(messages)
             response = SupportTicket.model_validate_json(response_text)
             messages.append({"role": "assistant", "content": response.model_dump_json()})
             return response
        except ValidationError as err:
            last_error = err
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Tu respuesta no es válida:\n{err}\nDevolvé SOLO el JSON corregido."})
            continue            
    raise ExtractionFailed("Failed to extract ticket after maximum repair attempts.") from last_error