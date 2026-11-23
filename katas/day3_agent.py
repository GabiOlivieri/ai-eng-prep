"""Día 3 (45 min): loop de agente con tool calling, sin framework.

`ChatFn` recibe (messages, tools) y devuelve un dict con la forma de
`choices[0].message` de la API de OpenAI:
  {"role": "assistant", "content": str | None,
   "tool_calls": [{"id": str, "type": "function",
                   "function": {"name": str, "arguments": str (JSON)}}] | None}
"""
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

Message = dict[str, Any]
ChatFn = Callable[[list[Message], list[dict[str, Any]]], Awaitable[Message]]
ToolFn = Callable[..., Awaitable[Any]]


class Tool(BaseModel):
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def to_openai(self) -> dict[str, Any]:
        # TODO: {"type": "function", "function": {name, description, parameters: schema}}
        raise NotImplementedError


class MaxStepsExceeded(Exception):
    pass


async def run_agent(user_msg: str, chat: ChatFn, tools: list[Tool], max_steps: int = 6) -> str:
    # TODO: loop; ejecutar tool_calls en paralelo; validar args con args_model;
    # si una tool falla, devolver el error como contenido del mensaje "tool" (no romper);
    # tool desconocida -> mensaje de error al modelo; superar max_steps -> MaxStepsExceeded.
    raise NotImplementedError
