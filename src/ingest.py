#!/usr/bin/env python3
"""
ingest.py
---------
CLI script to run the full ingestion pipeline:
    Extract → Chunk → Embed → Index (FAISS)

Usage examples
--------------
# Local embeddings (default)
python ingest.py --connection-string "..." --container my-container

# Azure OpenAI embeddings
python ingest.py --connection-string "..." --container my-container \\
    --embedder azure \\
    --azure-endpoint https://<resource>.openai.azure.com/ \\
    --azure-api-key <key> \\
    --azure-deployment text-embedding-ada-002

# Custom chunking + output paths + rebuild index from scratch
python ingest.py --connection-string "..." --container my-container \\
    --chunk-size 800 --overlap 120 \\
    --index-path my.index --metadata-path my_meta.json \\
    --recreate --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src importable when the script is called from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from extract import DocumentExtractor
from chunk import TextChunker
from embed import EmbeddingClient, LocalEmbeddingClient
from index import LocalFaissIndexManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Run the full document ingestion pipeline: Extract → Chunk → Embed → Index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Azure Blob Storage ─────────────────────────────────────────────────
    storage = parser.add_argument_group("Azure Blob Storage")
    storage.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        metavar="CONN_STR",
        help="Azure Storage connection string. Falls back to AZURE_STORAGE_CONNECTION_STRING env var.",
    )
    storage.add_argument(
        "--container",
        default=os.getenv("AZURE_CONTAINER_NAME", "knowledge-base"),
        metavar="NAME",
        help="Blob container name.",
    )
    storage.add_argument(
        "--prefix",
        default="",
        metavar="PREFIX",
        help="Optional blob prefix to filter ingested blobs (e.g. 'manuals/').",
    )

    # ── Extraction / OCR ───────────────────────────────────────────────────
    ocr = parser.add_argument_group("Extraction / OCR")
    ocr.add_argument(
        "--tesseract-lang",
        default="eng",
        metavar="LANG",
        help="Tesseract language pack (e.g. 'eng', 'fra').",
    )
    ocr.add_argument(
        "--ocr-threshold",
        type=int,
        default=100,
        metavar="N",
        help="Min character count from direct PDF extraction before falling back to OCR.",
    )

    # ── Chunking ───────────────────────────────────────────────────────────
    chunking = parser.add_argument_group("Chunking")
    chunking.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        metavar="N",
        help="Target number of characters per chunk.",
    )
    chunking.add_argument(
        "--overlap",
        type=int,
        default=200,
        metavar="N",
        help="Number of overlapping characters between consecutive chunks.",
    )
    chunking.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        metavar="N",
        help="Chunks smaller than this are merged with the preceding chunk.",
    )

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding = parser.add_argument_group("Embedding")
    embedding.add_argument(
        "--embedder",
        choices=["local", "azure"],
        default="local",
        help="Embedding backend: 'local' uses sentence-transformers, 'azure' uses Azure OpenAI.",
    )
    embedding.add_argument(
        "--local-model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="Sentence-transformers model name (used when --embedder=local).",
    )
    embedding.add_argument(
        "--azure-endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT"),
        metavar="URL",
        help="Azure OpenAI endpoint URL (used when --embedder=azure).",
    )
    embedding.add_argument(
        "--azure-api-key",
        default=os.getenv("AZURE_OPENAI_KEY"),
        metavar="KEY",
        help="Azure OpenAI API key (used when --embedder=azure).",
    )
    embedding.add_argument(
        "--azure-deployment",
        default="text-embedding-ada-002",
        metavar="NAME",
        help="Azure OpenAI embedding deployment name (used when --embedder=azure).",
    )
    embedding.add_argument(
        "--azure-api-version",
        default="2024-02-01",
        metavar="VERSION",
        help="Azure OpenAI API version string (used when --embedder=azure).",
    )
    embedding.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Number of chunks per embedding API call (used when --embedder=azure).",
    )
    embedding.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Delay between embedding retries on transient errors (used when --embedder=azure).",
    )

    # ── FAISS Index ────────────────────────────────────────────────────────
    index = parser.add_argument_group("FAISS Index")
    index.add_argument(
        "--index-path",
        default="faiss.index",
        metavar="PATH",
        help="Output path for the FAISS index file.",
    )
    index.add_argument(
        "--metadata-path",
        default="faiss_metadata.json",
        metavar="PATH",
        help="Output path for the FAISS metadata JSON file.",
    )
    index.add_argument(
        "--recreate",
        action="store_true",
        help="Delete any existing index files and rebuild from scratch.",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed INFO-level logging from all modules.",
    )

    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    # Suppress noisy third-party loggers
    for noisy in (
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.core",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Validate required arguments ────────────────────────────────────────
    if not args.connection_string:
        parser.error(
            "A storage connection string is required. "
            "Pass --connection-string or set the AZURE_STORAGE_CONNECTION_STRING env var."
        )

    if args.embedder == "azure":
        missing = [
            flag
            for flag, val in [
                ("--azure-endpoint", args.azure_endpoint),
                ("--azure-api-key", args.azure_api_key),
            ]
            if not val
        ]
        if missing:
            parser.error(
                f"Azure embedder requires: {', '.join(missing)}. "
                "Pass them as arguments or set the corresponding env vars "
                "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY)."
            )

    # ── Step 1: Extract ────────────────────────────────────────────────────
    logger.info("[1/4] Extracting documents from container '%s'", args.container)

    extractor = DocumentExtractor(
        connection_string=args.connection_string,
        container_name=args.container,
        tesseract_lang=args.tesseract_lang,
        ocr_threshold=args.ocr_threshold,
    )
    documents = extractor.extract_all(prefix=args.prefix)

    if not documents:
        logger.warning("No supported documents found. Exiting.")
        sys.exit(0)

    logger.info("Extracted %d document(s)", len(documents))

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    logger.info(
        "[2/4] Chunking documents (chunk_size=%d, overlap=%d, min_chunk_size=%d)",
        args.chunk_size,
        args.overlap,
        args.min_chunk_size,
    )

    chunker = TextChunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
    )
    chunks = chunker.chunk_documents(documents)
    logger.info("Produced %d chunk(s) across %d document(s).", len(chunks), len(documents))

    # ── Step 3: Embed ──────────────────────────────────────────────────────
    if args.embedder == "local":
        logger.info("[3/4] Embedding with local model '%s'", args.local_model)
        embedder = LocalEmbeddingClient(model_name=args.local_model)
    else:
        logger.info("[3/4] Embedding with Azure OpenAI deployment '%s'", args.azure_deployment)
        embedder = EmbeddingClient(
            azure_endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
            deployment_name=args.azure_deployment,
            batch_size=args.batch_size,
            retry_delay_seconds=args.retry_delay,
        )

    chunks = embedder.embed_chunks(chunks)
    sample_emb = chunks[0].metadata["embedding"]
    logger.info(
        "Embedding done. Model: %s | Dim: %d",
        chunks[0].metadata["embedding_model"],
        len(sample_emb),
    )

    # ── Step 4: Index ──────────────────────────────────────────────────────
    logger.info("[4/4] Building FAISS index -> '%s'", args.index_path)

    index_manager = LocalFaissIndexManager(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        recreate=args.recreate,
    )
    index_manager.build(chunks)

    logger.info("Index saved. Total documents indexed: %d", index_manager.get_document_count())
    logger.info("Index file   : %s", args.index_path)
    logger.info("Metadata file: %s", args.metadata_path)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()