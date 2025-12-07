"""
Document Processor for Hedwiq Agent

Handles PDF parsing with coordinate extraction and text segmentation.
Key features:
- Extract text with bounding boxes from PDF using PyMuPDF
- Segment documents for retrieval
- Generate embeddings using Azure OpenAI

Usage:
    processor = PDFProcessor()
    pages = processor.parse_pdf("/path/to/file.pdf")
    segments = processor.segment_document(pages, document_id="doc-123")

    embedding_gen = EmbeddingGenerator()
    embeddings = await embedding_gen.generate_embeddings(segments)
"""

import os
import re
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

from schemas.documents import (
    BoundingBox,
    TextSpan,
    DocumentPage,
    DocumentSegment,
    MAX_SEGMENT_LENGTH,
)

logger = logging.getLogger("hedwiq-document-processor")


class PDFProcessor:
    """
    Process PDF documents with coordinate extraction for precise highlighting.

    Uses PyMuPDF (fitz) for parsing, which provides text-layer bounding boxes
    at the span level. These coordinates are stored with segments to enable
    accurate highlighting in the frontend.
    """

    def __init__(self):
        """Initialize PDF processor."""
        # Lazy import to avoid startup delay
        self._fitz = None

    def _get_fitz(self):
        """Lazy load PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF is required for PDF processing. "
                    "Install it with: pip install PyMuPDF"
                )
        return self._fitz

    def parse_pdf(self, file_path: str) -> List[DocumentPage]:
        """
        Extract text with bounding boxes from PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            List of DocumentPage objects with text and coordinates

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF
        """
        fitz = self._get_fitz()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}")

        pages = []

        try:
            for page_num, page in enumerate(doc, 1):
                # Get page dimensions
                rect = page.rect

                # Extract text with coordinates using "dict" mode
                text_dict = page.get_text("dict")

                text_spans = []
                full_text_parts = []

                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:  # Text block (not image)
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "")
                                bbox = span.get("bbox", [0, 0, 0, 0])

                                if text.strip():
                                    text_spans.append(TextSpan(
                                        text=text,
                                        page_number=page_num,
                                        bbox=BoundingBox(
                                            x0=bbox[0],
                                            y0=bbox[1],
                                            x1=bbox[2],
                                            y1=bbox[3]
                                        )
                                    ))
                                    full_text_parts.append(text)

                pages.append(DocumentPage(
                    page_number=page_num,
                    text=" ".join(full_text_parts),
                    text_spans=text_spans,
                    width=rect.width,
                    height=rect.height
                ))

        finally:
            doc.close()

        return pages

    def parse_pdf_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> List[DocumentPage]:
        """
        Extract text with bounding boxes from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Optional filename for error messages

        Returns:
            List of DocumentPage objects with text and coordinates
        """
        fitz = self._get_fitz()

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Failed to open PDF '{filename}': {e}")

        pages = []

        try:
            for page_num, page in enumerate(doc, 1):
                rect = page.rect
                text_dict = page.get_text("dict")

                text_spans = []
                full_text_parts = []

                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "")
                                bbox = span.get("bbox", [0, 0, 0, 0])

                                if text.strip():
                                    text_spans.append(TextSpan(
                                        text=text,
                                        page_number=page_num,
                                        bbox=BoundingBox(
                                            x0=bbox[0],
                                            y0=bbox[1],
                                            x1=bbox[2],
                                            y1=bbox[3]
                                        )
                                    ))
                                    full_text_parts.append(text)

                pages.append(DocumentPage(
                    page_number=page_num,
                    text=" ".join(full_text_parts),
                    text_spans=text_spans,
                    width=rect.width,
                    height=rect.height
                ))

        finally:
            doc.close()

        return pages

    def segment_document(
        self,
        pages: List[DocumentPage],
        document_id: str,
        max_segment_length: int = MAX_SEGMENT_LENGTH
    ) -> List[DocumentSegment]:
        """
        Split document into segments with bounding boxes.

        Segments are sized for optimal retrieval (not too small to lose context,
        not too large to reduce precision).

        Args:
            pages: List of DocumentPage objects
            document_id: ID of the parent document
            max_segment_length: Maximum characters per segment

        Returns:
            List of DocumentSegment objects with bounding boxes
        """
        segments = []

        for page in pages:
            paragraphs = self._split_paragraphs(page.text)

            for para in paragraphs:
                if not para.strip():
                    continue

                chunks = self._chunk_text(para, max_segment_length)

                for chunk in chunks:
                    if len(chunk.strip()) < 10:
                        continue

                    # Find bounding box for this chunk
                    bbox = self._find_chunk_bbox(chunk, page.text_spans)

                    segment = DocumentSegment(
                        document_id=document_id,
                        page_number=page.page_number,
                        section_title=self._detect_section_title(chunk),
                        content=chunk,
                        bbox=bbox
                    )
                    segments.append(segment)

        return segments

    def _find_chunk_bbox(
        self,
        chunk: str,
        text_spans: List[TextSpan]
    ) -> Optional[BoundingBox]:
        """
        Find bounding box that covers the chunk text.

        Uses fuzzy matching to handle whitespace differences between
        extracted text and original PDF spans.
        """
        if not text_spans:
            return None

        chunk_lower = chunk.lower()
        matching_spans = []

        # Find spans that overlap with this chunk
        for span in text_spans:
            span_text_lower = span.text.lower()
            # Check for any word overlap
            if any(
                word in chunk_lower
                for word in span_text_lower.split()
                if len(word) > 3
            ):
                matching_spans.append(span)

        if not matching_spans:
            return None

        # Compute union bounding box
        x0 = min(s.bbox.x0 for s in matching_spans)
        y0 = min(s.bbox.y0 for s in matching_spans)
        x1 = max(s.bbox.x1 for s in matching_spans)
        y1 = max(s.bbox.y1 for s in matching_spans)

        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks at sentence boundaries.

        Tries to preserve semantic units by splitting at sentence endings.
        """
        if len(text) <= max_length:
            return [text]

        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle sentences longer than max_length
                if len(sentence) > max_length:
                    # Split by words
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= max_length:
                            current_chunk = (current_chunk + " " + word).strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _detect_section_title(self, text: str) -> Optional[str]:
        """
        Detect section title from text.

        Looks for common patterns like all-caps lines or lines ending with ":".
        """
        lines = text.split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Check if first line looks like a title
        if len(first_line) < 100:
            # All caps (common for headings)
            if first_line.isupper() and len(first_line) > 3:
                return first_line

            # Ends with colon (section header)
            if first_line.endswith(":"):
                return first_line.rstrip(":")

            # Numbered section (e.g., "1.2 Introduction")
            if re.match(r'^[\d.]+\s+\w', first_line):
                return first_line

        return None

    def extract_title(self, pages: List[DocumentPage]) -> str:
        """
        Extract document title from first page.

        Uses heuristics to find the most likely title.
        """
        if not pages:
            return "Untitled Document"

        first_page = pages[0]

        # Try to find a title in the first few lines
        lines = first_page.text.split("\n")
        for line in lines[:10]:
            line = line.strip()
            # Skip empty lines and very short lines
            if len(line) < 5:
                continue
            # Skip page numbers
            if re.match(r'^\d+$', line):
                continue
            # Skip headers like "Page 1 of 10"
            if re.match(r'^Page \d+', line, re.IGNORECASE):
                continue
            # Return first substantial line as title
            if len(line) < 200:
                return line[:200]

        return "Untitled Document"


class EmbeddingGenerator:
    """
    Generate embeddings for document segments using Azure OpenAI.

    Uses text-embedding-3-large (3072 dimensions) for high-quality
    semantic search.
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initialize embedding generator.

        Args:
            model_name: Azure OpenAI embedding model deployment name
        """
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy load Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI SDK is required for embeddings. "
                    "Install it with: pip install openai"
                )

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")

            if not api_key or not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                    "environment variables are required for embeddings"
                )

            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

        return self._client

    def generate_embeddings(
        self,
        segments: List[DocumentSegment]
    ) -> np.ndarray:
        """
        Generate embeddings for all segments.

        Args:
            segments: List of DocumentSegment objects

        Returns:
            numpy array of shape (num_segments, embedding_dim)
        """
        if not segments:
            return np.array([])

        client = self._get_client()

        texts = [s.content for s in segments]

        # Azure OpenAI supports batch embedding
        # Process in batches of 100 to avoid rate limits
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                raise

        return np.array(all_embeddings)

    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (for queries).

        Args:
            text: Text to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        client = self._get_client()

        try:
            response = client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise


class DocumentSummarizer:
    """
    Generate document summaries for semantic matching.

    Uses Azure OpenAI to create concise summaries optimized
    for keyword and semantic matching.
    """

    SUMMARY_PROMPT = """Create a concise summary for semantic matching.

Document: {title}
Content (first 3000 chars): {content}

Provide a 100-150 word summary covering:
1. Main topics and themes
2. Key technical terms and concepts
3. Document type and purpose

Write as a single paragraph optimized for keyword matching."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize summarizer.

        Args:
            model_name: Azure OpenAI chat model deployment name
        """
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy load Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI SDK is required for summaries. "
                    "Install it with: pip install openai"
                )

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")

            if not api_key or not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                    "environment variables are required"
                )

            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

        return self._client

    async def generate_summary(self, title: str, content: str) -> str:
        """
        Generate document summary.

        Args:
            title: Document title
            content: Document content (will be truncated to 3000 chars)

        Returns:
            Summary string (100-150 words)
        """
        import asyncio

        client = self._get_client()
        content_preview = content[:3000]

        prompt = self.SUMMARY_PROMPT.format(
            title=title,
            content=content_preview
        )

        try:
            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Return truncated content as fallback
            return content[:500] + "..."

    def generate_summary_sync(self, title: str, content: str) -> str:
        """
        Synchronous version of generate_summary.

        Args:
            title: Document title
            content: Document content

        Returns:
            Summary string
        """
        client = self._get_client()
        content_preview = content[:3000]

        prompt = self.SUMMARY_PROMPT.format(
            title=title,
            content=content_preview
        )

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return content[:500] + "..."
