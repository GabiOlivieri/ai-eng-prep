import numpy as np

import pytest

from katas.day2_rag import (
    chunk_markdown,
    evaluate_retrieval,
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_fusion,
    top_k_cosine,
)

DOC = "# Intro\nHola.\n\n## Instalación\n" + ("pip install algo. " * 150) + "\n\n## Uso\nCorré el comando."


def test_chunking_respects_size_and_keeps_sections():
    chunks = chunk_markdown("guia.md", DOC, max_chars=1200, overlap=200)
    assert all(len(c.text) <= 1200 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert any("Corré el comando" in c.text for c in chunks)
    assert len(chunks) >= 4  # Intro, Instalación partida en >=2, Uso


def test_top_k_cosine_orders_by_similarity():
    m = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=float)
    assert top_k_cosine(m, np.array([1.0, 0.1]), k=2) == [0, 2]


def test_rrf():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused.index("c") < fused.index("b")


def test_retrieval_metrics():
    assert recall_at_k(["x", "a", "y"], {"a", "b"}, k=2) == 0.5
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["x"], {"a"}) == 0.0


# Ejemplo trabajado: 4 relevantes, 2 de ellos en el top-5.
RETRIEVED = ["d7", "a", "d9", "b", "d3", "c", "d"]
RELEVANT = {"a", "b", "c", "d"}


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, RELEVANT, k=5) == 1.0
    assert hit_at_k(RETRIEVED, RELEVANT, k=1) == 0.0  # d7 no es relevante
    assert hit_at_k(["x", "y"], {"a"}, k=10) == 0.0


def test_recall_at_k_worked_example():
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == 0.5
    assert recall_at_k(RETRIEVED, RELEVANT, k=7) == 1.0
    assert recall_at_k(RETRIEVED, RELEVANT, k=1) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(2 / 5)
    # Si el retriever devuelve menos de k, igual se divide por k.
    assert precision_at_k(["a"], RELEVANT, k=5) == pytest.approx(1 / 5)


def test_mrr_worked_example():
    assert mrr(RETRIEVED, RELEVANT) == 0.5  # primer relevante en posición 2


def test_metrics_ignore_duplicates_and_empty_gold():
    # Un retriever con bug que repite ids no debe inflar el recall.
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5
    # Query sin relevantes: definimos 0.0 en vez de ZeroDivisionError.
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_evaluate_retrieval_averages_over_queries():
    runs = [
        (["a", "x", "y"], {"a"}),        # hit, recall 1, mrr 1
        (["x", "y", "b"], {"b", "c"}),   # hit@3, recall 0.5, mrr 1/3
        (["x", "y", "z"], {"q"}),        # miss total
    ]
    m = evaluate_retrieval(runs, k=3)
    assert m["hit@k"] == pytest.approx(2 / 3)
    assert m["recall@k"] == pytest.approx(0.5)
    assert m["precision@k"] == pytest.approx((1 / 3 + 1 / 3 + 0) / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
