"""Día 2 (45 + 40 min): RAG desde cero, búsqueda híbrida con RRF.

Solución de referencia. El pipeline completo es:
  documentos -> chunk_markdown -> embeddings (matriz N x D) + índice BM25
  query -> top_k_cosine (denso) + BM25 (léxico) -> reciprocal_rank_fusion -> contexto al LLM
  y para medir si el retriever sirve: hit@k, recall@k, precision@k, MRR sobre un gold set.
"""
import hashlib
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    source: str
    text: str


# Divide justo ANTES de cada heading ("lookahead"), así el heading queda pegado a su sección.
_HEADING = re.compile(r"(?m)^(?=#{1,6} )")


def chunk_markdown(source: str, text: str, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Corta por headings y, si una sección es larga, en ventanas de max_chars con overlap.

    - step = max_chars - overlap: cada ventana arranca `step` caracteres después de la anterior,
      así los últimos `overlap` caracteres se repiten en la siguiente (no se pierde contexto
      en el borde). Si overlap >= max_chars, step <= 0 y el loop no avanza: error.
    - El id es determinístico (source + índice + hash del texto): re-indexar el mismo documento
      produce los mismos ids, lo que permite upserts idempotentes en el vector store.
    """
    if overlap >= max_chars:
        raise ValueError("overlap tiene que ser menor que max_chars (si no, step <= 0)")
    step = max_chars - overlap
    sections = [s.strip() for s in _HEADING.split(text) if s.strip()]

    chunks: list[Chunk] = []
    for section in sections:
        if len(section) <= max_chars:
            pieces = [section]
        else:
            pieces = []
            for start in range(0, len(section), step):
                pieces.append(section[start : start + max_chars])
                if start + max_chars >= len(section):
                    break  # esta ventana ya llegó al final: evitamos un último pedazo redundante
        for piece in pieces:
            digest = hashlib.sha1(piece.encode()).hexdigest()[:8]
            chunks.append(Chunk(id=f"{source}#{len(chunks)}-{digest}", source=source, text=piece))
    return chunks


def top_k_cosine(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Índices de las k filas más parecidas a `query`, de mayor a menor similitud coseno.

    - coseno = producto punto de vectores normalizados (norma 1).
    - El epsilon evita dividir por cero si hay un vector nulo.
    - np.argpartition es O(N): deja los k mejores al final sin ordenar todo (O(N log N)).
      Después ordenamos solo esos k.
    """
    if k <= 0 or len(matrix) == 0:
        return []
    k = min(k, len(matrix))
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    q = query / (np.linalg.norm(query) + 1e-12)
    scores = m @ q
    top = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-scores[top])]]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fusiona varios rankings (ej. denso + BM25) sin tener que calibrar sus scores.

    score(doc) = sum(1 / (k + rank)), rank empieza en 1. k=60 es el valor del paper original:
    amortigua la diferencia entre el puesto 1 y el 2, y premia aparecer en varias listas.
    `sorted` es estable: ante empates queda primero el que apareció antes.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# --- Métricas de retrieval (bloque 2, ~25 min) -------------------------------
# retrieved: ids en el orden que devolvió el retriever (posición 0 = el mejor).
# relevant: ids que un humano marcó como correctos para esa query (el "gold set").
#
# Ejemplo trabajado (el de los tests):
#   retrieved = [d7, a, d9, b, d3, c, d]      relevant = {a, b, c, d}
#   top-5     = [d7, a, d9, b, d3]  -> relevantes en el top-5: {a, b}
#   hit@5       = 1.0        (hay al menos uno)
#   recall@5    = 2/4 = 0.5  (encontré 2 de los 4 que existen)
#   precision@5 = 2/5 = 0.4  (de los 5 que mostré, 2 sirven)
#   MRR         = 1/2        (el primer relevante, "a", está en la posición 2)
#
# Decisiones de borde (decilas en voz alta en la entrevista):
#   - relevant vacío -> 0.0 (no hay nada que encontrar; evitamos ZeroDivisionError).
#   - k > len(retrieved) -> precision igual divide por k (mostrar menos resultados no premia).
#   - ids repetidos en retrieved -> se cuentan una vez (usamos sets), no inflan las métricas.


def _hits_in_top_k(retrieved: list[str], relevant: set[str], k: int) -> set[str]:
    return set(retrieved[:k]) & relevant


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if _hits_in_top_k(retrieved, relevant, k) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(_hits_in_top_k(retrieved, relevant, k)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(_hits_in_top_k(retrieved, relevant, k)) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    # Técnicamente esto es el "reciprocal rank" de UNA query; el "mean" sale de promediar
    # sobre muchas queries en evaluate_retrieval.
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def evaluate_retrieval(
    runs: list[tuple[list[str], set[str]]], k: int
) -> dict[str, float]:
    """Promedio (macro) de cada métrica sobre todas las queries."""
    if not runs:
        return {"hit@k": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0}
    n = len(runs)
    return {
        "hit@k": sum(hit_at_k(r, g, k) for r, g in runs) / n,
        "recall@k": sum(recall_at_k(r, g, k) for r, g in runs) / n,
        "precision@k": sum(precision_at_k(r, g, k) for r, g in runs) / n,
        "mrr": sum(mrr(r, g) for r, g in runs) / n,
    }
