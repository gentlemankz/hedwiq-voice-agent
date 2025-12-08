"""
Hedwiq Agent Schemas

This module contains Pydantic models and enums for data validation and serialization.
"""

from .insights import Insight, InsightType, INSIGHT_ICONS
from .documents import (
    DocumentStatus,
    BoundingBox,
    TextSpan,
    DocumentPage,
    DocumentSegment,
    UploadedDocument,
    DocumentReference,
    RetrievalCandidate,
    MAX_DOCUMENTS_PER_ROOM,
    MAX_SEGMENTS_PER_DOCUMENT,
    MAX_SEGMENT_LENGTH,
    DOCUMENT_TTL_HOURS,
    DEDUPE_TTL_MINUTES,
    MIN_SEGMENT_WORDS,
    MIN_SEGMENT_DURATION,
    RRF_K,
)

__all__ = [
    # Insights
    "Insight",
    "InsightType",
    "INSIGHT_ICONS",
    # Documents
    "DocumentStatus",
    "BoundingBox",
    "TextSpan",
    "DocumentPage",
    "DocumentSegment",
    "UploadedDocument",
    "DocumentReference",
    "RetrievalCandidate",
    "MAX_DOCUMENTS_PER_ROOM",
    "MAX_SEGMENTS_PER_DOCUMENT",
    "MAX_SEGMENT_LENGTH",
    "DOCUMENT_TTL_HOURS",
    "DEDUPE_TTL_MINUTES",
    # Hybrid retrieval constants
    "MIN_SEGMENT_WORDS",
    "MIN_SEGMENT_DURATION",
    "RRF_K",
]
