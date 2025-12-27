"""
Insight Schemas for Luframe Agent

Defines the data models for meeting insights extracted by the AI agent.

Improvements:
- Timestamp now in milliseconds for frontend compatibility
- Minimum confidence threshold raised to 0.75
- Added min_length validation for content
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import time


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


class InsightType(str, Enum):
    """Types of insights that can be extracted from meeting conversations."""

    IDEA = "idea"
    PROBLEM = "problem"
    SOLUTION = "solution"
    RISK = "risk"
    INSIGHT = "insight"
    HYPOTHESIS = "hypothesis"
    ACTION_ITEM = "action_item"
    OPEN_QUESTION = "open_question"


class Insight(BaseModel):
    """
    Represents an insight extracted from meeting conversation.

    Attributes:
        type: The category of insight (idea, problem, solution, etc.)
        content: The actual insight content (8-20 words recommended)
        speaker: The identity of the speaker who mentioned this
        speaker_name: Display name of the speaker
        confidence: Confidence score from 0.0 to 1.0 (0.75+ recommended)
        transcript_ref: Reference to the transcript segment ID
        timestamp: Unix timestamp in MILLISECONDS when the insight was detected
    """

    type: InsightType
    content: str = Field(..., min_length=20, max_length=500, description="The insight content (8+ words)")
    speaker: Optional[str] = Field(None, description="Speaker identity token")
    speaker_name: Optional[str] = Field(None, description="Speaker display name")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    transcript_ref: Optional[str] = Field(None, description="Reference to transcript segment")
    timestamp: int = Field(default_factory=get_timestamp_ms, description="Timestamp in milliseconds")

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Ensure content has at least 8 words."""
        word_count = len(v.split())
        if word_count < 8:
            raise ValueError(f"Content must have at least 8 words, got {word_count}")
        return v

    class Config:
        use_enum_values = True


# Icon mappings for frontend display
INSIGHT_ICONS = {
    InsightType.IDEA: "lightbulb",
    InsightType.PROBLEM: "alert-triangle",
    InsightType.SOLUTION: "check-circle",
    InsightType.RISK: "alert-circle",
    InsightType.INSIGHT: "search",
    InsightType.HYPOTHESIS: "flask-conical",
    InsightType.ACTION_ITEM: "clipboard-list",
    InsightType.OPEN_QUESTION: "help-circle",
}
