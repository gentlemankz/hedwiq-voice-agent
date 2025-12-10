"""
Hedwiq Agent Schemas

This module contains Pydantic models and enums for data validation and serialization.
"""

from .insights import Insight, InsightType, INSIGHT_ICONS
from .agenda import (
    AgendaItemStatus,
    AgendaProgressType,
    AgendaItem,
    Agenda,
    AgendaProgressUpdate,
    TopicAnalysisResult,
    MAX_AGENDA_ITEMS,
    MIN_SEGMENTS_FOR_ANALYSIS,
    MIN_ANALYSIS_INTERVAL_SECONDS,
    ANALYSIS_DELAY_SECONDS,
    MIN_CONFIDENCE_FOR_COMPLETION,
    MIN_CONFIDENCE_FOR_SOFT_SIGNAL,
    MAX_TRANSCRIPT_WINDOW,
    MIN_SEGMENTS_SINCE_TOPIC_START,
    TRANSITION_COOLDOWN_SECONDS,
    AGENDA_STATUS_ICONS,
)
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
    # Agenda
    "AgendaItemStatus",
    "AgendaProgressType",
    "AgendaItem",
    "Agenda",
    "AgendaProgressUpdate",
    "TopicAnalysisResult",
    "MAX_AGENDA_ITEMS",
    "MIN_SEGMENTS_FOR_ANALYSIS",
    "MIN_ANALYSIS_INTERVAL_SECONDS",
    "ANALYSIS_DELAY_SECONDS",
    "MIN_CONFIDENCE_FOR_COMPLETION",
    "MIN_CONFIDENCE_FOR_SOFT_SIGNAL",
    "MAX_TRANSCRIPT_WINDOW",
    "MIN_SEGMENTS_SINCE_TOPIC_START",
    "TRANSITION_COOLDOWN_SECONDS",
    "AGENDA_STATUS_ICONS",
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
