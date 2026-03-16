"""
extract.py
----------
Extracts text and metadata from documents stored in Azure Blob Storage.
Supports PDF (digital + scanned via OCR), Markdown, and plain text formats.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContainerClient
import fitz  # PyMuPDF
import markdown
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExtractedDocument:
    """Holds the raw text and metadata extracted from a single document."""

    blob_name: str                     # Full path in the container, e.g. /manuals/deviceA.pdf
    source_url: str                    # Public/SAS URL of the blob
    content_type: str                  # "pdf" | "markdown" | "txt"
    text: str                          # Extracted plain text
    page_count: Optional[int] = None   # PDFs only
    title: Optional[str] = None        # Inferred from filename or first heading
    category: Optional[str] = None     # Top-level folder, e.g. "manuals"
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extraction_method: str = "direct"  # "direct" | "ocr"
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Stable identifier derived from the blob path."""
        return PurePosixPath(self.blob_name).stem.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class DocumentExtractor:
    """
    Pulls documents from Azure Blob Storage and extracts their text content.

    Parameters
    ----------
    connection_string : str
        Azure Storage connection string.
    container_name : str
        Name of the blob container.
    tesseract_lang : str
        OCR language pack code used by Tesseract.
    ocr_threshold : int
        Minimum character count from direct PDF extraction; below this value
        the document is considered scanned and re-processed with OCR.
    """

    def __init__(
        self,
        connection_string: str,
        container_name: str,
        tesseract_lang: str = "eng",
        ocr_threshold: int = 100,
    ) -> None:
        self.container_name = container_name
        self.ocr_threshold = ocr_threshold
        self.tesseract_lang = tesseract_lang

        self._blob_service: BlobServiceClient = BlobServiceClient.from_connection_string(
            connection_string
        )
        self._container: ContainerClient = self._blob_service.get_container_client(
            container_name
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, prefix: str = "") -> list[ExtractedDocument]:
        """
        List every supported blob under *prefix* and extract its content.

        Returns a list of ExtractedDocument objects (failures are logged and
        skipped rather than raising).
        """
        blobs = [
            b
            for b in self._container.list_blobs(name_starts_with=prefix)
            if self._is_supported(b.name)
        ]
        logger.debug("Found %d supported blobs under prefix '%s'", len(blobs), prefix)

        documents: list[ExtractedDocument] = []
        for blob in blobs:
            try:
                doc = self.extract_one(blob.name)
                documents.append(doc)
                logger.debug("Extracted '%s' (%d chars)", blob.name, len(doc.text))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to extract '%s': %s", blob.name, exc)

        return documents

    def extract_one(self, blob_name: str) -> ExtractedDocument:
        """Download and extract a single blob by its full path."""
        blob_client = self._container.get_blob_client(blob_name)
        properties = blob_client.get_blob_properties()
        raw_bytes: bytes = blob_client.download_blob().readall()

        ext = PurePosixPath(blob_name).suffix.lower()
        category = PurePosixPath(blob_name).parts[0] if "/" in blob_name else ""
        if ext == ".pdf":
            text, page_count, method = self._extract_pdf(raw_bytes)
        elif ext in (".md", ".markdown"):
            text = self._extract_markdown(raw_bytes)
            page_count, method = None, "direct"
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
            page_count, method = None, "direct"

        title = self._infer_title(blob_name, text, ext)

        return ExtractedDocument(
            blob_name=blob_name,
            source_url=blob_client.url,
            content_type=ext.lstrip("."),
            text=text,
            page_count=page_count,
            title=title,
            category=category,
            extraction_method=method,
            metadata={
                "size_bytes": properties.size,
                "last_modified": properties.last_modified.isoformat()
                if properties.last_modified
                else None,
                "content_type": properties.content_settings.content_type,
                "etag": properties.etag,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pdf(self, data: bytes) -> tuple[str, int, str]:
        """
        Attempt direct text extraction with PyMuPDF.
        Fall back to Tesseract OCR if the result is too sparse.
        """
        text, page_count, images = self._pymupdf_extract(data)
        method = "direct"
        if len(text.strip()) < self.ocr_threshold and len(images) > 0:
            logger.debug("Sparse PDF text (%d chars); switching to OCR.", len(text))
            try:
                ocr_text = self._ocr_extract(data)
                if ocr_text.strip():
                    text = ocr_text
                    method = "ocr"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tesseract OCR failed; keeping direct extraction: %s", exc)
        return text, page_count, method

    @staticmethod
    def _pymupdf_extract(data: bytes) -> tuple[str, int]:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        images = []
        for page in doc:
            pages.append(page.get_text("text"))
            images.extend(page.get_images(full=True))
        return "\n".join(pages), doc.page_count, images

    def _ocr_extract(self, data: bytes) -> str:
        """Run local Tesseract OCR over rendered PDF pages."""
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            pages: list[str] = []
            matrix = fitz.Matrix(2, 2)  # 2x render improves OCR accuracy on scans.
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(
                    pytesseract.image_to_string(image, lang=self.tesseract_lang)
                )
            return "\n".join(pages)
        finally:
            doc.close()

    @staticmethod
    def _extract_markdown(data: bytes) -> str:
        """Strip HTML tags produced by the markdown→HTML conversion."""
        raw = data.decode("utf-8", errors="replace")
        html = markdown.markdown(raw)
        clean = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _is_supported(name: str) -> bool:
        return PurePosixPath(name).suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}

    @staticmethod
    def _infer_title(blob_name: str, text: str, ext: str) -> str:
        """Use first heading from Markdown/text, otherwise derive from filename."""
        if ext in (".md", ".markdown"):
            match = re.search(r"^#+\s+(.+)", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        stem = PurePosixPath(blob_name).stem
        return stem.replace("_", " ").replace("-", " ").title()
