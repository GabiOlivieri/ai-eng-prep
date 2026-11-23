"""Día 1, ejercicio 2 (40 min): extracción estructurada con loop de reparación."""
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, EmailStr, Field

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
    raise NotImplementedError
