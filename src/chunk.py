"""
chunk.py
--------
Splits extracted document text into retrieval-friendly chunks.

Strategy
~~~~~~~~
1. Prefer splitting on natural boundaries: headings, paragraphs, sentences.
2. Apply a sliding window with configurable overlap so context is not lost
   at chunk edges.
3. Attach per-chunk metadata (source document, position, page estimate).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from extract import ExtractedDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single retrieval unit derived from an ExtractedDocument."""

    chunk_id: str                      # Globally unique identifier
    doc_id: str                        # Parent document identifier
    blob_name: str                     # Source blob path
    source_url: str                    # Source blob URL
    title: str                         # Parent document title
    category: str                      # Folder-level category
    text: str                          # Chunk text
    chunk_index: int                   # Zero-based position in the document
    total_chunks: int                  # Total chunks for this document (set after splitting)
    char_start: int                    # Character offset in the full document text
    char_end: int                      # Character offset end
    estimated_page: Optional[int]      # Best-effort page estimate (PDF only)
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (≈ 4 chars per token)."""
        return max(1, len(self.text) // 4)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Splits document text into overlapping chunks using LangChain's recursive
    character splitter with a retrieval-oriented separator hierarchy.

    Parameters
    ----------
    chunk_size : int
        Target number of *characters* per chunk (default 1 200, ~300 tokens).
    overlap : int
        Number of characters to repeat from the end of the previous chunk at
        the start of the next (default 200).
    min_chunk_size : int
        Chunks smaller than this are merged with the preceding chunk.
    """

    # Ordered list of boundary patterns (strongest to weakest)
    _SEPARATORS: list[str] = [
        r"\n#{1,6}\s",          # Markdown headings
        r"\n\n+",               # Paragraph breaks
        r"\n",                  # Line breaks
        r"(?<=[.!?])\s+",       # Sentence boundaries
        r"\s+",                 # Whitespace fallback
    ]

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            separators=self._SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            keep_separator=True,
            is_separator_regex=True,
            add_start_index=True,
            strip_whitespace=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ExtractedDocument) -> list[DocumentChunk]:
        """Return a list of DocumentChunk objects for a single document."""
        raw_chunks = self._split(doc.text)

        # Merge tiny trailing chunks
        raw_chunks = self._merge_small(raw_chunks)

        chunks: list[DocumentChunk] = []
        for idx, (text, char_start, char_end) in enumerate(raw_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc.doc_id,
                    blob_name=doc.blob_name,
                    source_url=doc.source_url,
                    title=doc.title or doc.doc_id,
                    category=doc.category or "",
                    text=text.strip(),
                    chunk_index=idx,
                    total_chunks=0,            # Back-filled below
                    char_start=char_start,
                    char_end=char_end,
                    estimated_page=self._estimate_page(
                        char_start, len(doc.text), doc.page_count
                    ),
                    metadata={**doc.metadata, "extraction_method": doc.extraction_method},
                )
            )

        # Back-fill total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, docs: list[ExtractedDocument]) -> list[DocumentChunk]:
        """Chunk a collection of documents and return a flat list."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[tuple[str, int, int]]:
        """Return (chunk_text, char_start, char_end) tuples for the input text."""
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        split_documents = self._splitter.create_documents([text])
        chunks: list[tuple[str, int, int]] = []
        search_start = 0

        for split_document in split_documents:
            chunk_text = split_document.page_content
            if not chunk_text.strip():
                continue

            start_index = split_document.metadata.get("start_index")
            if not isinstance(start_index, int):
                start_index = text.find(chunk_text, search_start)
                if start_index == -1:
                    start_index = text.find(chunk_text)
                if start_index == -1:
                    start_index = search_start

            end_index = start_index + len(chunk_text)
            chunks.append((chunk_text, start_index, end_index))
            search_start = max(start_index + 1, end_index - self.overlap)

        return chunks

    def _merge_small(
        self, chunks: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Merge chunks below min_chunk_size into the previous one."""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for text, cs, ce in chunks[1:]:
            if len(text.strip()) < self.min_chunk_size:
                prev_text, prev_cs, _ = merged[-1]
                merged[-1] = (prev_text + " " + text, prev_cs, ce)
            else:
                merged.append((text, cs, ce))
        return merged

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_page(
        char_start: int, total_chars: int, page_count: Optional[int]
    ) -> Optional[int]:
        if not page_count or total_chars == 0:
            return None
        ratio = char_start / total_chars
        return max(1, round(ratio * page_count))
