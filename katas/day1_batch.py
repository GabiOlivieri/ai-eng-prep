"""Día 1, ejercicio 1 (30 min): batch async con concurrencia limitada y reintentos.

Solución de referencia. Ideas clave para explicar en la entrevista:
- `asyncio.Semaphore` limita cuántas llamadas hay "en vuelo" a la vez (rate limits del proveedor).
- `asyncio.gather` devuelve los resultados en el MISMO orden que las corrutinas que le pasaste,
  aunque terminen en otro orden. Por eso no hace falta reordenar nada.
- El backoff (sleep) se hace FUERA del semáforo: si dormís adentro, ocupás un cupo sin hacer nada.
- Backoff exponencial con jitter: base * 2**intento * random(0.5, 1.5) evita que todos los
  reintentos peguen juntos (thundering herd).
- Solo se reintenta lo que es reintentable (429, timeout). Un 400 o un bug no se reintenta.
"""
import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchResult:
    outputs: list[str]          # en el MISMO orden que los prompts
    p50_ms: float
    p95_ms: float


class RetryableError(Exception):
    """Simula un 429 o un timeout."""


def percentile(values: list[float], pct: float) -> float:
    """Percentil por "nearest rank": simple, sin dependencias y fácil de explicar.

    Ojo: `statistics.quantiles` necesita al menos 2 datos y por defecto interpola;
    con pocas muestras conviene algo predecible como esto.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))  # rank empieza en 1
    return ordered[rank - 1]


async def run_batch(
    prompts: list[str],
    call: Callable[[str], Awaitable[str]],
    max_concurrency: int = 5,
    max_retries: int = 3,
    base_delay: float = 0.01,
) -> BatchResult:
    sem = asyncio.Semaphore(max_concurrency)
    latencies_ms: list[float] = []  # una entrada por intento (incluye los que fallaron)

    async def one(prompt: str) -> str:
        # max_retries = cantidad TOTAL de intentos (el test con max_retries=2 espera rendirse).
        for attempt in range(max_retries):
            try:
                async with sem:  # el cupo se toma solo mientras se llama al modelo
                    start = time.perf_counter()
                    try:
                        return await call(prompt)
                    finally:
                        latencies_ms.append((time.perf_counter() - start) * 1000)
            except RetryableError:
                if attempt == max_retries - 1:
                    raise  # sin más intentos: propagamos el error original
                delay = base_delay * (2**attempt) * random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)  # fuera del semáforo
        raise AssertionError("inalcanzable")  # para el type checker

    # gather preserva el orden. Si una tarea falla, gather propaga la excepción.
    # (Alternativa: return_exceptions=True para no perder los resultados buenos.)
    outputs = await asyncio.gather(*(one(p) for p in prompts))
    return BatchResult(
        outputs=list(outputs),
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
    )
