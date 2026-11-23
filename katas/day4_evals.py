"""Día 4 (45 min): harness de evals con chequeos determinísticos + LLM-as-judge."""
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


async def run_evals(
    cases: list[EvalCase], system: SystemFn, judge: JudgeFn, max_concurrency: int = 4
) -> tuple[list[CaseResult], float]:
    # TODO: por caso, chequeos determinísticos; el juez solo si should_refuse es False.
    # passed = todos los chequeos ok y judge.passed (si hubo juez). Devolver (resultados, pass_rate).
    raise NotImplementedError
