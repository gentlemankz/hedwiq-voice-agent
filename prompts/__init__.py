"""
Hedwiq Agent Prompts

This module contains LLM prompts for insight extraction, analysis,
and document reference detection.
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
]
