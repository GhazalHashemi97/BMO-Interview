"""
embed.py
--------
Generates vector embeddings for document chunks using Azure OpenAI.

Features
~~~~~~~~
- Small fixed-size batching for chunk embeddings.
- Optional local sentence-transformers fallback (no Azure dependency for dev).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from openai import AzureOpenAI, APIStatusError, RateLimitError
from chunk import DocumentChunk
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EmbeddingClient for Azure OpenAI
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """
    Minimal Azure OpenAI embeddings client.

    Parameters
    ----------
    azure_endpoint : str
        e.g. ``https://<resource>.openai.azure.com/``
    api_key : str
        Azure OpenAI API key.
    api_version : str
        API version string, e.g. ``"2024-02-01"``.
    deployment_name : str
        Azure deployment name for the embedding model.
    batch_size : int
        Maximum number of chunks per API call.
    retry_delay_seconds : float
        Delay between retries for transient errors.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-01",
        deployment_name: str = "text-embedding-ada-002",
        batch_size: Optional[int] = 64,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment_name
        self._batch_size = batch_size
        self._retry_delay_seconds = retry_delay_seconds
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Embed chunks in place and return the input list."""
        if not chunks:
            return chunks

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embed_texts([c.text for c in batch])

            for chunk, vector in zip(batch, vectors):
                chunk.metadata["embedding"] = vector
                chunk.metadata["embedding_model"] = self._deployment
                chunk.metadata["embedding_dim"] = len(vector)

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_texts([query])[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(2):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._deployment,
                )
                items = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in items]
            except (RateLimitError, APIStatusError):
                if attempt == 1:
                    raise
                logger.warning("Embedding request failed, retrying once in %.1fs", self._retry_delay_seconds)
                time.sleep(self._retry_delay_seconds)

        raise RuntimeError("Embedding request failed after single retry.")  # pragma: no cover


# ---------------------------------------------------------------------------
# Using Local Embeddings for Development
# ---------------------------------------------------------------------------

class LocalEmbeddingClient:
    """
    Sentence-transformers-based embedder for local development.
    Drop-in replacement for EmbeddingClient.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        texts = [c.text for c in chunks]
        vectors = self._model.encode(texts, show_progress_bar=True).tolist()
        for chunk, vector in zip(chunks, vectors):
            chunk.metadata["embedding"] = vector
            chunk.metadata["embedding_model"] = self._model_name
            chunk.metadata["embedding_dim"] = len(vector)
        return chunks

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([query])[0].tolist()
