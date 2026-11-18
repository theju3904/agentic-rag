"""Embedding model: local sentence-transformers or remote GPU API.

Centralizing the encode call in one class means every retriever, the golden
dataset indexer, and the agentic query rewriter share one loaded model
instance and one normalization policy: embeddings are always L2-normalized,
so cosine similarity equals a plain dot product, matching both Qdrant's
COSINE distance and pgvector's `<=>` operator.

Set REMOTE_EMBEDDING_URL in .env to offload inference to a GPU HTTP server
(see embed_server_final.py). The remote server must expose POST /embed
accepting {"texts": [...], "model": "<name>"} and returning
{"embeddings": [[...]], "dimensions": N}.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Shared interface for local and remote embedding models.

    Both `EmbeddingModel` (sentence-transformers in-process) and
    `RemoteEmbeddingModel` (GPU HTTP server) satisfy this structurally, so
    consumers can accept either without a concrete-class dependency. L2
    normalization is part of the contract: `encode()` always returns unit
    vectors so cosine similarity equals a dot product.
    """

    dimensions: int

    def encode(
        self, texts: list[str], batch_size: int = ..., show_progress: bool = ...
    ) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...


class EmbeddingModel:
    """Encodes text into L2-normalized dense vectors with a local bi-encoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        embedding_dimensions = self._model.get_sentence_embedding_dimension()
        if embedding_dimensions is None:
            raise ValueError(f"Could not determine embedding dimensionality for '{model_name}'")
        self.dimensions: int = embedding_dimensions

    def encode(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Encode `texts` into a `(len(texts), dimensions)` L2-normalized array."""
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class RemoteEmbeddingModel:
    """Encodes text by calling a remote GPU HTTP embedding server.

    The server must expose:
      POST /embed  body: {"texts": [...], "model": "..."}
                   response: {"embeddings": [[...]], "dimensions": N}
      GET  /dims?model=...  response: {"dimensions": N}
    """

    def __init__(self, base_url: str, model_name: str, remote_batch_size: int = 512) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._remote_batch_size = remote_batch_size
        self.dimensions: int = self._fetch_dimensions()
        logger.info(
            "RemoteEmbeddingModel: %s @ %s  dim=%d", model_name, base_url, self.dimensions
        )

    def _fetch_dimensions(self) -> int:
        from urllib.parse import quote
        url = f"{self._base_url}/dims?model={quote(self._model_name)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return int(data["dimensions"])

    def _call_remote(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps(
            {"texts": texts, "model": self._model_name, "batch_size": self._remote_batch_size}
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                return np.array(data["embeddings"], dtype=np.float32)
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
                if attempt < 3:
                    wait = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(
                        "Embed request failed (attempt %d/4): %s — retrying in %ds",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    def encode(
        self, texts: list[str], batch_size: int = 512, show_progress: bool = False
    ) -> np.ndarray:
        """Encode `texts` via the remote server, chunking into `batch_size` requests."""
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        parts = [self._call_remote(chunk) for chunk in chunks]
        return np.vstack(parts).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_embedding_model(
    model_name: str, remote_url: str | None = None
) -> EmbeddingModelProtocol:
    """Return a local or remote embedding model depending on whether a URL is set."""
    if remote_url:
        return RemoteEmbeddingModel(base_url=remote_url, model_name=model_name)
    return EmbeddingModel(model_name)
