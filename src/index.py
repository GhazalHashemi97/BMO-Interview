"""
index.py — Local FAISS vector index: build, save, reload, and search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import faiss


from chunk import DocumentChunk

logger = logging.getLogger(__name__)


def _to_doc(chunk: DocumentChunk) -> dict:
    return {
        "id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "blob_name": chunk.blob_name,
        "source_url": chunk.source_url,
        "title": chunk.title,
        "category": chunk.category,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "estimated_page": chunk.estimated_page,
        "extraction_method": chunk.metadata.get("extraction_method", "direct"),
    }


class LocalFaissIndexManager:
    """FAISS-backed vector index: build, save, reload, and search."""

    def __init__(
        self,
        index_path: str | Path = "faiss.index",
        metadata_path: str | Path = "faiss_metadata.json",
        recreate: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._index = None
        self._docs: list[dict] = []
        if recreate:
            self.index_path.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build the FAISS index from embedded chunks and save to disk."""
        vectors = np.array([c.metadata["embedding"] for c in chunks], dtype="float32")
        faiss.normalize_L2(vectors)
        self._index = faiss.IndexFlatIP(vectors.shape[1]) #get only the dimension of the vectors
        self._index.add(vectors)
        self._docs = [_to_doc(c) for c in chunks]
        self.save()
        logger.debug("FAISS index built. Number of chunks: %d", len(chunks))

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("Index is empty — call build() first.")
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(json.dumps({"documents": self._docs}, indent=2))

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if faiss is None:
            raise ImportError("Run: pip install faiss-cpu")
        self._index = faiss.read_index(str(self.index_path))
        self._docs = json.loads(self.metadata_path.read_text())["documents"]
        logger.info("Loaded FAISS index (%d docs).", len(self._docs))

    def search(
        self,
        query_vector: list[float],
        top_n: int = 5,
    ) -> list[dict]:
        """Return the top_n nearest docs by cosine similarity."""
        if self._index is None:
            raise RuntimeError("Index not loaded — call load() or build() first.")
        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        k = min(top_n, len(self._docs))
        scores, indices = self._index.search(q, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self._docs[idx])
            doc["@search.score"] = float(score)
            results.append(doc)
        return results

    def get_document_count(self) -> int:
        return len(self._docs)

    def get_documents(self) -> list[dict]:
        return [dict(d) for d in self._docs]
