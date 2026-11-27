"""Día 3 (45 min): loop de agente con tool calling, sin framework.

`ChatFn` recibe (messages, tools) y devuelve un dict con la forma de
`choices[0].message` de la API de OpenAI:
  {"role": "assistant", "content": str | None,
   "tool_calls": [{"id": str, "type": "function",
                   "function": {"name": str, "arguments": str (JSON)}}] | None}

Solución de referencia. Ideas clave:
- Un agente es un while: llamar al modelo; si pide tools, ejecutarlas y devolverle los
  resultados como mensajes role="tool" (con el tool_call_id correspondiente); si no pide
  tools, su content es la respuesta final.
- El mensaje del assistant con los tool_calls TIENE que ir al historial antes de los
  mensajes "tool": la API valida que cada tool_call_id responda a un pedido previo.
- Varias tool_calls en un mismo turno son independientes: se ejecutan en paralelo con gather.
  gather preserva el orden, así que los mensajes "tool" quedan en el orden de los pedidos.
- Los errores (JSON inválido, args que no validan, tool inexistente, excepción de la tool)
  NO rompen el loop: vuelven al modelo como texto para que pueda corregirse o avisar.
- max_steps es el freno de seguridad contra loops infinitos (y contra gastar tokens).
- `system` (opcional): con modelos chicos, un system prompt que diga "completá TODO lo que
  pide el usuario antes de responder" hace la diferencia entre encadenar tools o frenar en la
  primera (lo vimos con qwen3:8b en el test live).
"""
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

Message = dict[str, Any]
ChatFn = Callable[[list[Message], list[dict[str, Any]]], Awaitable[Message]]
ToolFn = Callable[..., Awaitable[Any]]


class Tool(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class MaxStepsExceeded(Exception):
    pass


async def _execute(call: dict[str, Any], registry: dict[str, Tool]) -> Message:
    """Ejecuta UNA tool_call y devuelve el mensaje role="tool". Nunca lanza."""
    name = call["function"]["name"]
    tool = registry.get(name)
    try:
        if tool is None:
            raise LookupError(f"la tool '{name}' no existe. Tools disponibles: {list(registry)}")
        args = tool.args_model.model_validate_json(call["function"].get("arguments") or "{}")
        result = await tool.fn(**args.model_dump())
        content = json.dumps(result, ensure_ascii=False, default=str)
    except ValidationError as err:
        content = f"Error: argumentos inválidos para '{name}': {err}"
    except Exception as err:  # noqa: BLE001 -- a propósito: cualquier falla vuelve al modelo
        content = f"Error al ejecutar '{name}': {type(err).__name__}: {err}"
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


DEFAULT_SYSTEM = (
    "Sos un agente de soporte. Usá las tools para resolver TODO lo que pide el usuario: si el "
    "pedido tiene varios pasos o acciones condicionales (\"si pasa X, hacé Y\"), revisá el "
    "resultado de cada tool y ejecutá las acciones que correspondan ANTES de dar la respuesta "
    "final. No digas que hiciste algo que no hiciste con una tool."
)


async def run_agent(
    user_msg: str,
    chat: ChatFn,
    tools: list[Tool],
    max_steps: int = 6,
    system: str | None = DEFAULT_SYSTEM,
) -> str:
    registry = {t.name: t for t in tools}
    specs = [t.to_openai() for t in tools]
    messages: list[Message] = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user_msg})

    for _ in range(max_steps):
        reply = await chat(messages, specs)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return reply.get("content") or ""

        messages.append(
            {"role": "assistant", "content": reply.get("content"), "tool_calls": tool_calls}
        )
        results = await asyncio.gather(*(_execute(c, registry) for c in tool_calls))
        messages.extend(results)

    raise MaxStepsExceeded(f"el agente no terminó en {max_steps} pasos")
