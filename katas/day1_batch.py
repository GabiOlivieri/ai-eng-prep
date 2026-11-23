"""Día 1, ejercicio 1 (30 min): batch async con concurrencia limitada y reintentos."""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


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
    raise NotImplementedError
