"""Día 2 (45 + 40 min): RAG desde cero, búsqueda híbrida con RRF."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    source: str
    text: str


def chunk_markdown(source: str, text: str, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    # TODO: cortar por headings (#, ##) y, si una sección es larga, por tamaño con overlap.
    raise NotImplementedError


def top_k_cosine(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    # TODO: normalizar, producto punto, np.argpartition, ordenar los k mejores de mayor a menor.
    raise NotImplementedError


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    # TODO: score(doc) = sum(1 / (k + rank)) con rank empezando en 1; devolver ids ordenados.
    raise NotImplementedError


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    raise NotImplementedError
