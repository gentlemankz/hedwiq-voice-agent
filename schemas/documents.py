"""
Document Schemas for Hedwiq Agent

Defines the data models for document reference feature.
Documents are uploaded by admins and used for real-time reference detection
during meetings.

Key Features:
- BoundingBox support for precise PDF highlighting
- Room-scoped document management
- Segment-level indexing for retrieval
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import time
import uuid


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def generate_doc_id() -> str:
    """Generate unique document ID."""
    return f"doc-{uuid.uuid4().hex[:8]}"


def generate_segment_id() -> str:
    """Generate unique segment ID."""
    return f"seg-{uuid.uuid4().hex[:8]}"


def generate_ref_id() -> str:
    """Generate unique reference ID."""
    return f"ref-{uuid.uuid4().hex[:8]}"


class DocumentStatus(str, Enum):
    """Status of document processing."""
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class BoundingBox(BaseModel):
    """
    PDF coordinates for highlighting a region on a page.

    Coordinates are in PDF units (points), where origin is bottom-left.
    Frontend should convert to screen coordinates based on page dimensions.
    """
    x0: float = Field(..., description="Left edge X coordinate")
    y0: float = Field(..., description="Bottom edge Y coordinate")
    x1: float = Field(..., description="Right edge X coordinate")
    y1: float = Field(..., description="Top edge Y coordinate")

    @field_validator("x1")
    @classmethod
    def x1_greater_than_x0(cls, v: float, info) -> float:
        """Ensure x1 > x0."""
        if "x0" in info.data and v <= info.data["x0"]:
            raise ValueError("x1 must be greater than x0")
        return v

    @field_validator("y1")
    @classmethod
    def y1_greater_than_y0(cls, v: float, info) -> float:
        """Ensure y1 > y0."""
        if "y0" in info.data and v <= info.data["y0"]:
            raise ValueError("y1 must be greater than y0")
        return v

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }


class TextSpan(BaseModel):
    """
    A span of text with its PDF coordinates.

    Used during PDF parsing to track where each text fragment
    appears on the page.
    """
    text: str
    page_number: int
    bbox: BoundingBox


class DocumentPage(BaseModel):
    """
    Represents a single page from a PDF document.

    Contains both the extracted text and coordinate data
    for precise highlighting.
    """
    page_number: int
    text: str = Field(..., description="Full text content of the page")
    text_spans: List[TextSpan] = Field(default_factory=list, description="Text with coordinates")
    width: float = Field(..., description="Page width in points")
    height: float = Field(..., description="Page height in points")


class DocumentSegment(BaseModel):
    """
    A searchable segment of a document with coordinates.

    Documents are split into segments for efficient retrieval.
    Each segment has its own embedding and can be independently
    matched against speech.
    """
    id: str = Field(default_factory=generate_segment_id)
    document_id: str
    page_number: int
    section_title: Optional[str] = None
    content: str = Field(..., min_length=10, description="Segment text content")
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "content": self.content,
            "bbox": self.bbox.to_dict() if self.bbox else None,
        }


class UploadedDocument(BaseModel):
    """
    Metadata for an uploaded document.

    Documents are scoped to rooms and cleaned up after TTL expires.
    """
    id: str = Field(default_factory=generate_doc_id)
    room_id: str = Field(..., description="LiveKit room ID for scoping")
    filename: str = Field(..., max_length=255)
    title: str = Field(..., max_length=500)
    summary: str = Field(default="", max_length=2000)
    page_count: int = Field(..., ge=1)
    status: DocumentStatus = DocumentStatus.PROCESSING
    uploaded_at: int = Field(default_factory=get_timestamp_ms)
    uploaded_by: str = Field(..., description="User ID who uploaded")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "room_id": self.room_id,
            "filename": self.filename,
            "title": self.title,
            "summary": self.summary,
            "page_count": self.page_count,
            "status": self.status.value,
            "uploaded_at": self.uploaded_at,
            "uploaded_by": self.uploaded_by,
        }

    class Config:
        use_enum_values = True


class DocumentReference(BaseModel):
    """
    A reference from speech to document content.

    Created when the system detects that a speaker is referencing
    content from an uploaded document.
    """
    id: str = Field(default_factory=generate_ref_id)
    document_id: str
    section_id: str = Field(..., description="For deduplication")
    page_number: int = Field(..., ge=1)
    section_title: Optional[str] = None
    matched_text: str = Field(..., min_length=10, max_length=500, description="Evidence span from document")
    bbox: Optional[BoundingBox] = None
    context: str = Field(..., min_length=10, max_length=200, description="Why this is a match")
    confidence: float = Field(..., ge=0.0, le=1.0)
    transcript_ref: Optional[str] = Field(None, description="Reference to transcript segment")
    timestamp: int = Field(default_factory=get_timestamp_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "matched_text": self.matched_text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "context": self.context,
            "confidence": self.confidence,
            "transcript_ref": self.transcript_ref,
            "timestamp": self.timestamp,
        }


# Storage limits (enforced by PersistentDocumentStore)
MAX_DOCUMENTS_PER_ROOM = 10
MAX_SEGMENTS_PER_DOCUMENT = 500
MAX_SEGMENT_LENGTH = 500  # characters
DOCUMENT_TTL_HOURS = 24
DEDUPE_TTL_MINUTES = 5

# Pre-filter thresholds for hybrid retrieval
MIN_SEGMENT_WORDS = 6
MIN_SEGMENT_DURATION = 1.2  # seconds
RRF_K = 60  # Reciprocal Rank Fusion constant


class RetrievalCandidate(BaseModel):
    """
    A candidate document segment returned by hybrid retrieval.

    Contains the segment data along with retrieval scores for ranking.
    Used as input to the LLM alignment step.
    """
    segment_id: str
    document_id: str
    page_number: int
    section_title: Optional[str] = None
    content: str
    score: float = Field(..., ge=0.0, description="Combined RRF score")
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "segment_id": self.segment_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "content": self.content,
            "score": self.score,
            "bbox": self.bbox.to_dict() if self.bbox else None,
        }


# Document type icons for frontend display
DOCUMENT_TYPE_ICONS = {
    "pdf": "file-text",
}
