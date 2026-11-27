"""Día 2 (45 + 40 min): RAG desde cero, búsqueda híbrida con RRF."""
from dataclasses import dataclass

import numpy as np

import re


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    source: str
    text: str


def chunk_markdown(source: str, text: str, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    # TODO: cortar por headings (#, ##) y, si una sección es larga, por tamaño con overlap.
    if overlap >= max_chars:
        raise ValueError(...)          # ¿por qué? pensá qué valdría step
    step = ...
    sections = [s.strip() for s in re.split(r"(?m)^(?=#{1,6} )", text) if s.strip()]

    chunks: list[Chunk] = []
    for section in sections:
        if len(section) <= max_chars:
            pieces = [section]
        else:
            pieces = []
            for start in range(0, len(section), step):
                pieces.append(...)
                if start + max_chars >= len(section):
                    break
        for piece in pieces:
            chunks.append(Chunk(id=..., source=source, text=piece))
    return chunks


def top_k_cosine(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    # TODO: normalizar, producto punto, np.argpartition, ordenar los k mejores de mayor a menor.
    raise NotImplementedError


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    # TODO: score(doc) = sum(1 / (k + rank)) con rank empezando en 1; devolver ids ordenados.
    raise NotImplementedError


# --- Métricas de retrieval (bloque 2, ~25 min) -------------------------------
# retrieved: ids en el orden que devolvió el retriever (posición 0 = el mejor).
# relevant: ids que un humano marcó como correctos para esa query (el "gold set").
# Pensá siempre: ¿qué pasa si relevant está vacío? ¿y si k > len(retrieved)?


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # TODO: 1.0 si al menos un relevante aparece en el top-k, si no 0.0.
    raise NotImplementedError


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # TODO: de todos los relevantes, ¿qué fracción quedó dentro del top-k?
    raise NotImplementedError


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # TODO: de los k que mostraste, ¿qué fracción es relevante? (dividí por k, no por len(retrieved))
    raise NotImplementedError


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    # TODO: 1 / posición del PRIMER relevante (posición empieza en 1); 0.0 si no hay ninguno.
    raise NotImplementedError


def evaluate_retrieval(
    runs: list[tuple[list[str], set[str]]], k: int
) -> dict[str, float]:
    # TODO: promediar cada métrica sobre todas las queries.
    # Devolver {"hit@k": ..., "recall@k": ..., "precision@k": ..., "mrr": ...}
    raise NotImplementedError
