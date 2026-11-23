from katas.day4_evals import EvalCase, JudgeVerdict, SystemOutput, run_evals

CASES = [
    EvalCase(id="1", question="¿Cómo instalo?", expected_source="install.md"),
    EvalCase(id="2", question="¿Precio de Bitcoin?", should_refuse=True),
    EvalCase(id="3", question="¿Cómo deployo?", expected_source="deploy.md"),
]


async def system(q: str) -> SystemOutput:
    if "Bitcoin" in q:
        return SystemOutput(answer="No tengo esa info.", refused=True)
    if "instalo" in q:
        return SystemOutput(answer="pip install", sources=["install.md"])
    return SystemOutput(answer="Ni idea", sources=["faq.md"])  # fuente equivocada


judge_calls: list[str] = []


async def judge(case: EvalCase, out: SystemOutput) -> JudgeVerdict:
    judge_calls.append(case.id)
    return JudgeVerdict(score=4, reasoning="ok", passed=True)


async def test_run_evals():
    judge_calls.clear()
    results, rate = await run_evals(CASES, system, judge)
    by_id = {r.case_id: r for r in results}
    assert by_id["1"].passed and by_id["2"].passed
    assert not by_id["3"].passed and not by_id["3"].source_ok
    assert rate == 2 / 3
    assert "2" not in judge_calls  # no se juzgan los casos que deben rechazarse
