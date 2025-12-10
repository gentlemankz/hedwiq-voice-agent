"""
Hedwiq Agent Prompts

This module contains LLM prompts for insight extraction, analysis,
document reference detection, and agenda tracking.
"""

from .insight_extraction import (
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    INSIGHT_EXTRACTION_USER_TEMPLATE,
)

from .document_reference import (
    DOCUMENT_ALIGNMENT_SYSTEM_PROMPT,
    DOCUMENT_ALIGNMENT_USER_TEMPLATE,
    format_alignment_prompt,
    MIN_ALIGNMENT_CONFIDENCE,
    HIGH_CONFIDENCE_THRESHOLD,
    ALIGNMENT_TIMEOUT_SECONDS,
    ALIGNMENT_MAX_RETRIES,
)

from .agenda_tracking import (
    AGENDA_TRACKING_SYSTEM_PROMPT,
    AGENDA_TRACKING_USER_TEMPLATE,
    AGENDA_START_DETECTION_PROMPT,
    AGENDA_COMPLETION_CHECK_PROMPT,
    format_agenda_items,
    format_next_topic_info,
    format_tracking_prompt,
)

__all__ = [
    # Insight extraction
    "INSIGHT_EXTRACTION_SYSTEM_PROMPT",
    "INSIGHT_EXTRACTION_USER_TEMPLATE",
    # Document reference (Phase 3)
    "DOCUMENT_ALIGNMENT_SYSTEM_PROMPT",
    "DOCUMENT_ALIGNMENT_USER_TEMPLATE",
    "format_alignment_prompt",
    "MIN_ALIGNMENT_CONFIDENCE",
    "HIGH_CONFIDENCE_THRESHOLD",
    "ALIGNMENT_TIMEOUT_SECONDS",
    "ALIGNMENT_MAX_RETRIES",
    # Agenda tracking
    "AGENDA_TRACKING_SYSTEM_PROMPT",
    "AGENDA_TRACKING_USER_TEMPLATE",
    "AGENDA_START_DETECTION_PROMPT",
    "AGENDA_COMPLETION_CHECK_PROMPT",
    "format_agenda_items",
    "format_next_topic_info",
    "format_tracking_prompt",
]
