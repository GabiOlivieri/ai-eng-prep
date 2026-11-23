# ai-eng-prep (starter)

Esqueleto para el plan de 5 días. Los archivos de `katas/` tienen firmas y TODOs:
la idea es que los completes **a mano, sin autocompletado de IA**, con timer.
Los tests de `tests/` son el criterio de "terminado" de cada ejercicio y usan fakes,
así que corren sin modelo ni red.

```bash
uv sync
cp .env.example .env
ollama pull qwen3:8b && ollama pull nomic-embed-text
uv run pytest tests/test_day1.py      # va a fallar hasta que implementes la kata
```

| Día | Archivo | Tiempo |
| --- | --- | --- |
| 1 | `katas/day1_batch.py`, `katas/day1_extract.py` | 30 + 40 min |
| 2 | `katas/day2_rag.py` | 45 + 40 min |
| 3 | `katas/day3_agent.py` | 45 min |
| 4 | `katas/day4_evals.py` | 45 min |
| 5 | `katas/day5_guardrails.py` | 40 min |

`src/supportbot/` es el mini-proyecto acumulativo; arranca solo con `config.py`.
