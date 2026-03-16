"""
search.py — Local FAISS search: vector (cosine similarity), keyword (BM25),
and hybrid (vector + BM25 via RRF + cross-encoder semantic re-ranking).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional  # retained for SearchResult.estimated_page

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from embed import EmbeddingClient, LocalEmbeddingClient
from index import LocalFaissIndexManager
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    rank: int
    chunk_id: str
    doc_id: str
    blob_name: str
    source_url: str
    title: str
    category: str
    text: str
    chunk_index: int
    total_chunks: int
    estimated_page: Optional[int]
    score: float


@dataclass
class SearchResponse:
    query: str
    mode: SearchMode
    top_n: int
    results: list[SearchResult]


class LocalSearchEngine:
    """Local search: keyword (BM25), vector (cosine via FAISS), and hybrid (BM25 + vector
    via RRF followed by cross-encoder semantic re-ranking)."""

    def __init__(
        self,
        index_manager: LocalFaissIndexManager,
        embedder: EmbeddingClient | LocalEmbeddingClient,
        cross_encoder_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._index_manager = index_manager
        self._embedder = embedder
        self._cross_encoder = None
        if cross_encoder_model:
            try:
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("httpx").setLevel(logging.ERROR)
                self._cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load cross-encoder '%s': %s", cross_encoder_model, exc)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[SearchMode] = SearchMode.HYBRID,
        _RRF_K: int = 60
    ) -> SearchResponse:
        if mode == SearchMode.VECTOR:
            raw = self._vector_search(query, top_k)
        elif mode == SearchMode.KEYWORD:
            raw = self._bm25_search(query, top_k)
        else:  # HYBRID: BM25 + vector (RRF) + cross-encoder semantic re-ranking
            raw = self._hybrid_search(query, top_k, _RRF_K)
        results = [self._doc_to_result(rank, item) for rank, item in enumerate(raw, start=1)]
        return SearchResponse(query=query, mode=mode, top_n=top_k, results=results)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        return self._index_manager.search(self._embedder.embed_query(query), top_n=top_k)

    def _bm25_search(self, query: str, top_n: int) -> list[dict]:
        docs = self._index_manager.get_documents()
        if not docs:
            return []
        tokenize = lambda t: [w.lower() for w in word_tokenize(t) if w.isalnum()]
        query_terms = tokenize(query)
        if not query_terms:
            return []
        corpus = [tokenize(f"{d.get('title','')} {d.get('text','')}") for d in docs]
        scores = BM25Okapi(corpus).get_scores(query_terms)
        results = []
        for score, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True):
            if score == 0.0:
                break
            candidate = dict(doc)
            candidate["@search.score"] = float(score)
            results.append(candidate)
            if len(results) >= top_n:
                break
        return results

    def _hybrid_search(self, query: str, top_n: int, _RRF_K: int) -> list[dict]:
        k = top_n * 4
        # Stage 1 — keyword (BM25)
        fused: dict[str, dict] = {}
        for rank, item in enumerate(self._bm25_search(query, k), start=1):
            cur = fused.setdefault(item["id"], dict(item))
            cur["@search.score"] = cur.get("@search.score", 0.0) + 1.0 / (_RRF_K + rank)
        # Stage 2 — dense vector (FAISS)
        for rank, item in enumerate(self._vector_search(query, k), start=1):
            cur = fused.setdefault(item["id"], dict(item))
            cur["@search.score"] = cur.get("@search.score", 0.0) + 1.0 / (_RRF_K + rank)
        candidates = sorted(fused.values(), key=lambda d: d["@search.score"], reverse=True)[:top_n]
        # Stage 3 — semantic re-ranking (cross-encoder)
        if self._cross_encoder is not None:
            candidates = self._semantic_rerank(query, candidates)
        return candidates

    def _semantic_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score candidates with a cross-encoder and return them sorted by that score."""
        pairs = [
            (query, f"{d.get('title', '')} {d.get('text', '')}")
            for d in candidates
        ]
        scores = self._cross_encoder.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["@search.score"] = float(score)
        return sorted(candidates, key=lambda d: d["@search.score"], reverse=True)

    @staticmethod
    def _doc_to_result(rank: int, item: dict) -> SearchResult:
        return SearchResult(
            rank=rank,
            chunk_id=item["id"],
            doc_id=item["doc_id"],
            blob_name=item["blob_name"],
            source_url=item["source_url"],
            title=item["title"],
            category=item.get("category", ""),
            text=item["text"],
            chunk_index=item.get("chunk_index", 0),
            total_chunks=item.get("total_chunks", 0),
            estimated_page=item.get("estimated_page"),
            score=item.get("@search.score", 0.0),
        )

