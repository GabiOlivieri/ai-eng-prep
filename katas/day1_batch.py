"""Día 1, ejercicio 1 (30 min): batch async con concurrencia limitada y reintentos."""
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import statistics as statics
import logging
import time
import asyncio

logging.basicConfig(level=logging.INFO)

@dataclass(frozen=True, slots=True)
class BatchResult:
    outputs: list[str]          # en el MISMO orden que los prompts
    p50_ms: float
    p95_ms: float


class RetryableError(Exception):
    """Simula un 429 o un timeout."""


async def run_batch(
    prompts: list[str],
    call: Callable[[str], Awaitable[str]],
    max_concurrency: int = 5,
    max_retries: int = 3,
    base_delay: float = 0.01,
) -> BatchResult:
    # TODO: Semaphore + gather, backoff exponencial ante RetryableError, latencias por llamada.
    latencies: list[float] = []
    sem = asyncio.Semaphore(max_concurrency)

    async def one (prompt: str) -> str:
        for x in range(max_retries):
            try: 
                async with sem:
                    t0 = time.perf_counter()
                    try:
                        return await call(prompt)
                    finally:
                        latencies.append((time.perf_counter() - t0) * 1000)
                        logging.info("Call for prompt: %s took %.2f ms", prompt, latencies[-1])
            except RetryableError as e:
                logging.info("RetryableError: %s. Retrying...", e)
                if x < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** x))  # Exponential backoff
                else:
                    logging.error("Max retries reached for prompt: %s. Raising exception.", prompt)
                    raise
            except Exception as e:
                logging.error("Unexpected error: %s. Not retrying.", e)
                raise

    tasks = [one(prompt) for prompt in prompts]
    outputs = await asyncio.gather(*tasks)
    s = sorted(latencies)
    p50_ms = statics.median(s) if s else 0
    p95_ms = statics.quantiles(s, n=100)[94] if len(s) >= 20 else 0
    return BatchResult(outputs=outputs, p50_ms=p50_ms, p95_ms=p95_ms)
