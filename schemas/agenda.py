"""
Agenda Schemas for Hedwiq Agent

Defines Pydantic models for meeting agenda tracking and progress detection.

The agent receives agenda from frontend via LiveKit text stream (hedwiq.agenda),
tracks progress through the predefined agenda items using LLM analysis,
and publishes progress updates (hedwiq.agenda_progress).
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import time


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


# Configuration Constants (matching frontend/docs/AGENDA_FEATURE_PLAN.md Section 10)
# Defined here first so they can be used in Field constraints below
MAX_AGENDA_ITEMS = 10
MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 500

# Analysis triggers - tuned for responsive detection
MIN_SEGMENTS_FOR_ANALYSIS = 2  # Reduced from 4 - analyze sooner
MIN_ANALYSIS_INTERVAL_SECONDS = 8.0  # Reduced from 15 - more frequent checks
ANALYSIS_DELAY_SECONDS = 2.0  # Reduced from 5 - faster response

# Confidence thresholds
MIN_CONFIDENCE_FOR_COMPLETION = 0.7  # Reduced from 0.8 - less conservative
MIN_CONFIDENCE_FOR_SOFT_SIGNAL = 0.6  # Reduced from 0.7

# Transcript window
MAX_TRANSCRIPT_WINDOW = 20  # Last N segments for analysis
MIN_SEGMENTS_SINCE_TOPIC_START = 2  # Reduced from 5 - allow earlier completion detection

# Deduplication
TRANSITION_COOLDOWN_SECONDS = 20.0  # Reduced from 60 - allow faster transitions


class AgendaItemStatus(str, Enum):
    """Status of an agenda item during meeting progression."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class AgendaProgressType(str, Enum):
    """Types of progress updates sent to frontend."""
    TOPIC_STARTED = "topic_started"
    TOPIC_COMPLETED = "topic_completed"
    TOPIC_CHANGE = "topic_change"
    AGENDA_COMPLETE = "agenda_complete"


class AgendaItem(BaseModel):
    """
    Single agenda item received from frontend.

    This matches the frontend AgendaItem interface from types/agenda.ts.
    """
    id: str = Field(..., description="Unique identifier for the item")
    title: str = Field(..., min_length=1, max_length=MAX_TITLE_LENGTH, description="Agenda item title")
    description: Optional[str] = Field(None, max_length=MAX_DESCRIPTION_LENGTH, description="Optional description")
    estimated_minutes: Optional[int] = Field(None, ge=1, le=120, description="Estimated duration")
    lead_by: Optional[str] = Field(None, description="Optional presenter/leader name")
    order: int = Field(..., ge=0, description="Order in agenda (0-indexed)")

    class Config:
        populate_by_name = True


class Agenda(BaseModel):
    """
    Full agenda received from frontend via LiveKit text stream.

    Sent to agent when user joins the meeting room.
    """
    id: str = Field(..., description="Unique agenda identifier")
    room_id: str = Field(..., description="Room this agenda belongs to")
    items: List[AgendaItem] = Field(default_factory=list, description="Ordered list of agenda items")

    @field_validator("items")
    @classmethod
    def validate_items_count(cls, v: List[AgendaItem]) -> List[AgendaItem]:
        """Ensure agenda has reasonable number of items."""
        if len(v) > MAX_AGENDA_ITEMS:
            raise ValueError(f"Agenda cannot have more than {MAX_AGENDA_ITEMS} items")
        return v

    class Config:
        populate_by_name = True


class AgendaProgressUpdate(BaseModel):
    """
    Progress update sent from agent to frontend.

    Published to hedwiq.agenda_progress topic when topic transitions are detected.
    """
    type: AgendaProgressType = Field(..., description="Type of progress update")
    agenda_id: str = Field(..., description="ID of the agenda being tracked")
    item_index: int = Field(..., ge=-1, description="Index of affected item (-1 for no current item)")
    status: AgendaItemStatus = Field(..., description="New status of the item")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Detection confidence")
    reason: Optional[str] = Field(None, max_length=200, description="Why this decision was made")
    transcript_ref: Optional[str] = Field(None, description="Transcript segment that triggered change")
    timestamp: int = Field(default_factory=get_timestamp_ms, description="Timestamp in milliseconds")

    class Config:
        use_enum_values = True


class TopicAnalysisResult(BaseModel):
    """
    LLM analysis result for topic progression detection.

    Used internally to parse and validate LLM responses.
    """
    current_topic_complete: bool = Field(..., description="Whether current topic appears complete")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence: str = Field(..., max_length=200, description="Brief evidence for the decision")
    next_topic_started: bool = Field(default=False, description="Whether next topic has begun")
    next_topic_index: Optional[int] = Field(None, ge=0, description="Index of next topic if started")

    @field_validator("evidence")
    @classmethod
    def validate_evidence_length(cls, v: str) -> str:
        """Ensure evidence is concise but meaningful."""
        if len(v) < 5:
            raise ValueError("Evidence must be at least 5 characters")
        return v


# Icon mappings for frontend display
AGENDA_STATUS_ICONS = {
    AgendaItemStatus.PENDING: "circle",
    AgendaItemStatus.IN_PROGRESS: "play",
    AgendaItemStatus.COMPLETED: "check",
}
