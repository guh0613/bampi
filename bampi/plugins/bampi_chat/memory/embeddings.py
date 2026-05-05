from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .search_text import extract_search_terms

DEFAULT_EMBEDDING_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


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


@dataclass(slots=True)
class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for OpenAI-compatible `/embeddings` endpoints."""

    provider: str = "openai-compatible"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 15.0
    user_agent: str = DEFAULT_EMBEDDING_USER_AGENT
    max_retries: int = 2
    retry_base_delay: float = 0.3
    dimensions: int = 0
    _client: Any | None = field(default=None, init=False, repr=False)

    def embed_text(self, text: str) -> list[float]:
        if not self.model:
            raise ValueError(
                "bampi_memory_embedding_model is required for openai-compatible embeddings"
            )
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                result = client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float",
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max(0, self.max_retries):
                    raise
                time.sleep(self.retry_base_delay * (2**attempt))
        else:
            raise last_error or RuntimeError("embedding request failed")
        if not result.data:
            return []
        vector = [float(value) for value in result.data[0].embedding]
        self.dimensions = len(vector)
        return normalize_vector(vector)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = normalize_openai_embedding_base_url(self.base_url)
            if self.user_agent:
                kwargs["default_headers"] = {"User-Agent": self.user_agent}
            self._client = OpenAI(**kwargs)
        return self._client


def normalize_openai_embedding_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def build_embedding_provider(
    *,
    provider: str = "",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 15.0,
    max_retries: int = 2,
) -> MemoryEmbeddingProvider:
    provider_key = provider.strip().lower().replace("_", "-")
    if provider_key in {"", "local", "local-hash", "hash"}:
        return LocalHashEmbeddingProvider(
            provider="local-hash",
            model=model or "local-hash-v1",
        )
    if provider_key in {"openai", "openai-compatible"}:
        if not model.strip():
            raise ValueError(
                "bampi_memory_embedding_model is required for openai-compatible embeddings"
            )
        return OpenAICompatibleEmbeddingProvider(
            provider=provider_key,
            model=model,
            api_key=api_key,
            base_url=normalize_openai_embedding_base_url(base_url),
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(
        "bampi_memory_embedding_provider must be one of: local-hash, openai-compatible"
    )


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
