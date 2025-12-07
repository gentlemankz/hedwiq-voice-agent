"""
Hedwiq Agent Prompts

This module contains LLM prompts for insight extraction and analysis.
"""

from .insight_extraction import (
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    INSIGHT_EXTRACTION_USER_TEMPLATE,
)

__all__ = [
    "INSIGHT_EXTRACTION_SYSTEM_PROMPT",
    "INSIGHT_EXTRACTION_USER_TEMPLATE",
]
