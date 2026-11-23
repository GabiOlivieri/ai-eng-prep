import asyncio
import json

import pytest

from katas.day1_batch import RetryableError, run_batch
from katas.day1_extract import ExtractionFailed, SupportTicket, extract_ticket


async def test_batch_keeps_order_limits_concurrency_and_retries():
    attempts: dict[str, int] = {}
    in_flight = 0
    peak = 0

    async def call(prompt: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            attempts[prompt] = attempts.get(prompt, 0) + 1
            if prompt == "p3" and attempts[prompt] <= 2:
                raise RetryableError("429")
            return prompt.upper()
        finally:
            in_flight -= 1

    prompts = [f"p{i}" for i in range(20)]
    result = await run_batch(prompts, call, max_concurrency=5)
    assert result.outputs == [p.upper() for p in prompts]
    assert peak <= 5
    assert attempts["p3"] == 3
    assert 0 < result.p50_ms <= result.p95_ms


async def test_batch_gives_up_after_max_retries():
    async def always_fails(_: str) -> str:
        raise RetryableError("429")

    with pytest.raises(RetryableError):
        await run_batch(["x"], always_fails, max_retries=2)


VALID = json.dumps(
    {"category": "billing", "priority": 4, "customer_email": "ana@example.com",
     "summary": "Le cobraron dos veces"}
)


async def test_extract_repairs_invalid_output():
    responses = iter(['{"category": "pagos", "priority": 9}', VALID])
    seen: list[list[dict[str, str]]] = []

    async def llm(messages):
        seen.append(messages)
        return next(responses)

    ticket = await extract_ticket("Me cobraron dos veces! ana@example.com", llm)
    assert ticket == SupportTicket.model_validate_json(VALID)
    assert len(seen) == 2
    # el segundo intento tiene que incluir el error de validación para que el modelo se corrija
    assert "priority" in json.dumps(seen[1])


async def test_extract_fails_after_max_repairs():
    async def llm(_):
        return "no es json"

    with pytest.raises(ExtractionFailed):
        await extract_ticket("hola", llm, max_repairs=2)
