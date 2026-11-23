import json

import pytest
from pydantic import BaseModel

from katas.day3_agent import MaxStepsExceeded, Tool, run_agent


class OrderArgs(BaseModel):
    order_id: str


class TicketArgs(BaseModel):
    summary: str
    priority: int


created: list[str] = []


async def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "delayed"}


async def create_ticket(summary: str, priority: int) -> dict:
    created.append(summary)
    return {"ticket_id": "T-1"}


async def broken(order_id: str) -> dict:
    raise RuntimeError("db caída")


TOOLS = [
    Tool(name="get_order_status", description="Estado de un pedido", args_model=OrderArgs,
         fn=get_order_status),
    Tool(name="create_ticket", description="Crea un ticket", args_model=TicketArgs, fn=create_ticket),
]


def call(id_: str, name: str, args: dict) -> dict:
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(*turns):
    it = iter(turns)
    log: list[list[dict]] = []

    async def chat(messages, tools):
        log.append(list(messages))
        return next(it)

    return chat, log


def test_tool_schema():
    spec = TOOLS[0].to_openai()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "get_order_status"
    assert "order_id" in spec["function"]["parameters"]["properties"]


async def test_agent_runs_tools_in_sequence():
    created.clear()
    chat, log = scripted(
        {"role": "assistant", "content": None,
         "tool_calls": [call("c1", "get_order_status", {"order_id": "123"})]},
        {"role": "assistant", "content": None,
         "tool_calls": [call("c2", "create_ticket", {"summary": "Pedido 123 demorado", "priority": 3})]},
        {"role": "assistant", "content": "Listo, abrí el ticket T-1.", "tool_calls": None},
    )
    answer = await run_agent("¿Cómo viene el pedido 123?", chat, TOOLS)
    assert answer == "Listo, abrí el ticket T-1."
    assert created == ["Pedido 123 demorado"]
    tool_msgs = [m for m in log[-1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


async def test_tool_errors_go_back_to_the_model():
    tools = [Tool(name="get_order_status", description="x", args_model=OrderArgs, fn=broken)]
    chat, log = scripted(
        {"role": "assistant", "content": None,
         "tool_calls": [call("c1", "get_order_status", {"order_id": "1"}),
                        call("c2", "no_existe", {})]},
        {"role": "assistant", "content": "No pude consultarlo.", "tool_calls": None},
    )
    assert await run_agent("estado?", chat, tools) == "No pude consultarlo."
    tool_msgs = [m for m in log[-1] if m["role"] == "tool"]
    assert "db caída" in tool_msgs[0]["content"]
    assert "no_existe" in tool_msgs[1]["content"]


async def test_max_steps():
    loop_turn = {"role": "assistant", "content": None,
                 "tool_calls": [call("c", "get_order_status", {"order_id": "1"})]}
    chat, _ = scripted(*[loop_turn] * 10)
    with pytest.raises(MaxStepsExceeded):
        await run_agent("x", chat, TOOLS, max_steps=3)
