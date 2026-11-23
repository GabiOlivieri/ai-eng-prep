import numpy as np

from katas.day2_rag import chunk_markdown, mrr, recall_at_k, reciprocal_rank_fusion, top_k_cosine

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
