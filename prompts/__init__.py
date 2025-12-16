"""
Hedwiq Agent Prompts

This module contains LLM prompts for insight extraction, analysis,
document reference detection, and action classification.
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

from .action_classification import (
    ACTION_CLASSIFICATION_SYSTEM_PROMPT,
    ACTION_CLASSIFICATION_USER_TEMPLATE,
    ACTION_BATCH_CLASSIFICATION_USER_TEMPLATE,
    format_classification_prompt,
    format_batch_classification_prompt,
    URGENCY_PATTERNS,
    MIN_CLASSIFICATION_CONFIDENCE,
    HIGH_CONFIDENCE_THRESHOLD as ACTION_HIGH_CONFIDENCE_THRESHOLD,
    CLASSIFICATION_TIMEOUT_SECONDS,
    CLASSIFICATION_MAX_RETRIES,
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
    # Action classification (Phase 1 of Real-Time Actions)
    "ACTION_CLASSIFICATION_SYSTEM_PROMPT",
    "ACTION_CLASSIFICATION_USER_TEMPLATE",
    "ACTION_BATCH_CLASSIFICATION_USER_TEMPLATE",
    "format_classification_prompt",
    "format_batch_classification_prompt",
    "URGENCY_PATTERNS",
    "MIN_CLASSIFICATION_CONFIDENCE",
    "ACTION_HIGH_CONFIDENCE_THRESHOLD",
    "CLASSIFICATION_TIMEOUT_SECONDS",
    "CLASSIFICATION_MAX_RETRIES",
]
