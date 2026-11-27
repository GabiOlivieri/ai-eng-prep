"""Día 4 (45 min): harness de evals con chequeos determinísticos + LLM-as-judge.

Solución de referencia. Ideas clave:
- Primero los chequeos baratos y determinísticos (¿citó la fuente esperada? ¿rechazó cuando
  debía?). Son reproducibles y no cuestan tokens.
- El juez (otro LLM) solo para lo subjetivo (¿la respuesta es correcta y completa?). No se
  juzgan los casos que deben rechazarse: ahí alcanza con el chequeo de `refused`.
- Concurrencia limitada con Semaphore, igual que en el día 1.
- Un caso que explota (el sistema o el juez lanzan) cuenta como fallado, no tira abajo
  toda la corrida: en una suite de 200 casos querés el reporte igual.
"""
import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    expected_facts: list[str] = []
    expected_source: str | None = None
    should_refuse: bool = False


class SystemOutput(BaseModel):
    answer: str
    sources: list[str] = []
    refused: bool = False


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str
    passed: bool


class CaseResult(BaseModel):
    case_id: str
    source_ok: bool
    refusal_ok: bool
    judge: JudgeVerdict | None
    passed: bool


SystemFn = Callable[[str], Awaitable[SystemOutput]]
JudgeFn = Callable[[EvalCase, SystemOutput], Awaitable[JudgeVerdict]]


async def _eval_case(case: EvalCase, system: SystemFn, judge: JudgeFn) -> CaseResult:
    try:
        out = await system(case.question)
        source_ok = case.expected_source is None or case.expected_source in out.sources
        refusal_ok = out.refused == case.should_refuse
        verdict = None if case.should_refuse else await judge(case, out)
    except Exception as err:  # noqa: BLE001
        failed = JudgeVerdict(score=1, reasoning=f"error: {err!r}", passed=False)
        return CaseResult(case_id=case.id, source_ok=False, refusal_ok=False, judge=failed,
                          passed=False)

    passed = source_ok and refusal_ok and (verdict is None or verdict.passed)
    return CaseResult(case_id=case.id, source_ok=source_ok, refusal_ok=refusal_ok,
                      judge=verdict, passed=passed)


async def run_evals(
    cases: list[EvalCase], system: SystemFn, judge: JudgeFn, max_concurrency: int = 4
) -> tuple[list[CaseResult], float]:
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(case: EvalCase) -> CaseResult:
        async with sem:
            return await _eval_case(case, system, judge)

    results = await asyncio.gather(*(guarded(c) for c in cases))
    pass_rate = sum(r.passed for r in results) / len(results) if results else 0.0
    return list(results), pass_rate
