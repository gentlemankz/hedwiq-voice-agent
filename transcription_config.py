"""
Shared STT configuration helpers for Luframe agents.

Centralizing these keeps the transcription-only agent and the
insight-enabled agent in sync and makes it easier to swap
providers or defaults.
"""

import os
from typing import Optional, List


def get_stt_model() -> str:
    """Deepgram model name."""
    return os.getenv("STT_MODEL", "nova-3")


def get_stt_language() -> str:
    """Language code; supports 'multi' for multilingual rooms."""
    return os.getenv("STT_LANGUAGE", "en-US")


def get_stt_keyterms() -> Optional[List[str]]:
    """Optional comma-separated keyterms to bias STT for proper nouns."""
    raw = os.getenv("STT_KEYTERMS", "")
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    return terms or None

