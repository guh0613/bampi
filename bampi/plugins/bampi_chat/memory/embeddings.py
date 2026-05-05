from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from .search_text import extract_search_terms


class MemoryEmbeddingProvider(Protocol):
    provider: str
    model: str
    dimensions: int

    def embed_text(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class LocalHashEmbeddingProvider:
    """Deterministic local vectorizer used for optional archive-level recall.

    This is deliberately dependency-free and offline. It is not a replacement
    for provider embeddings, but it gives the optional embedding path a stable
    implementation and ranking signal without making session disposal depend on
    network calls.
    """

    provider: str = "local-hash"
    model: str = "local-hash-v1"
    dimensions: int = 128

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * max(8, self.dimensions)
        for term in extract_search_terms(text, for_query=False):
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % len(vector)
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[index] += sign
        return normalize_vector(vector)


def normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(limit))
