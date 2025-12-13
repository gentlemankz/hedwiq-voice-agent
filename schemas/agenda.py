"""
Agenda Schemas for Hedwiq Agent - Phase 4 Implementation

Defines the data models for meeting agenda tracking.
These mirror the frontend types in frontend/types/agenda.ts and the database
schema in frontend/lib/db/schema.ts.

Key Features:
- AgendaItem and Agenda models matching database schema
- AgendaProgressEvent types for LiveKit communication
- AgendaStateAttribute for late joiner sync via participant attributes
- Stability/hysteresis tracking for topic detection
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import time


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


# Stop words for keyword extraction - shared across all usages
# FIX (R1+R2): Consolidate keyword extraction to single location
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "it", "its", "we", "our", "us", "about"
}


def extract_keywords(title: str, description: Optional[str] = None) -> List[str]:
    """
    Extract keywords from title and description for matching.

    FIX (R1+R2): Consolidate keyword extraction to single function.
    This replaces duplicate implementations in AgendaTracker and AgendaItem.

    Args:
        title: Item title
        description: Optional item description

    Returns:
        List of unique keywords (lowercase, stop words removed)
    """
    text = title.lower()
    if description:
        text += " " + description.lower()

    words = text.split()
    keywords = [
        w.strip(".,;:!?()[]{}\"'")
        for w in words
        if len(w) > 2 and w.lower() not in STOP_WORDS
    ]
    return list(set(keywords))


# ============================================================================
# Status Types
# ============================================================================

class AgendaStatus(str, Enum):
    """Overall agenda status."""
    DRAFT = "draft"       # Being created/edited in PreJoin (editable)
    ACTIVE = "active"     # Meeting in progress (locked, tracking progress)
    COMPLETED = "completed"  # Meeting ended (all items processed)


class AgendaItemStatus(str, Enum):
    """Individual agenda item status."""
    PENDING = "pending"           # Not yet started
    IN_PROGRESS = "in_progress"   # Currently being discussed
    COMPLETED = "completed"       # Discussion finished
    SKIPPED = "skipped"           # Intentionally skipped


# ============================================================================
# Core Data Models
# ============================================================================

class AgendaItem(BaseModel):
    """
    An individual agenda item (topic).
    Matches the database schema from `agenda_item` table.
    """
    id: str
    agenda_id: str = Field(alias="agendaId")
    title: str
    description: Optional[str] = None
    estimated_duration: Optional[int] = Field(None, alias="estimatedDuration")
    presenter: Optional[str] = None
    order_index: int = Field(alias="orderIndex")
    status: AgendaItemStatus = AgendaItemStatus.PENDING
    started_at: Optional[str] = Field(None, alias="startedAt")
    completed_at: Optional[str] = Field(None, alias="completedAt")
    actual_duration: Optional[int] = Field(None, alias="actualDuration")
    start_transcript_ref: Optional[str] = Field(None, alias="startTranscriptRef")
    end_transcript_ref: Optional[str] = Field(None, alias="endTranscriptRef")

    # Keywords extracted from title/description for fast matching
    keywords: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        use_enum_values = True

    def get_keywords(self) -> List[str]:
        """
        Extract keywords from title and description for matching.

        FIX (R1+R2): Now uses shared extract_keywords() function.
        """
        return extract_keywords(self.title, self.description)


class Agenda(BaseModel):
    """
    A meeting agenda with all its items.
    Matches the database schema from `agenda` table.
    """
    id: str
    room_id: str = Field(alias="roomId")
    created_by: str = Field(alias="createdBy")
    item_count: int = Field(alias="itemCount")
    status: AgendaStatus = AgendaStatus.DRAFT
    current_item_index: Optional[int] = Field(None, alias="currentItemIndex")
    version: int = 1
    meeting_started_at: Optional[str] = Field(None, alias="meetingStartedAt")
    meeting_ended_at: Optional[str] = Field(None, alias="meetingEndedAt")
    items: List[AgendaItem] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        use_enum_values = True


# ============================================================================
# LiveKit Event Types
# ============================================================================

class AgendaEventType(str, Enum):
    """Types of agenda progress events."""
    MEETING_STARTED = "meeting_started"
    MEETING_ENDED = "meeting_ended"
    TOPIC_STARTED = "topic_started"
    TOPIC_COMPLETED = "topic_completed"
    TOPIC_SKIPPED = "topic_skipped"
    AGENDA_SYNC = "agenda_sync"


@dataclass
class AgendaProgressEvent:
    """Base class for agenda progress events."""
    type: str
    timestamp: int = field(default_factory=get_timestamp_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "timestamp": self.timestamp,
        }


@dataclass
class MeetingStartedEvent(AgendaProgressEvent):
    """Event: Meeting started (first topic begins)."""
    type: str = field(default=AgendaEventType.MEETING_STARTED.value)


@dataclass
class MeetingEndedEvent(AgendaProgressEvent):
    """Event: Meeting ended (all topics completed or skipped)."""
    type: str = field(default=AgendaEventType.MEETING_ENDED.value)


@dataclass
class TopicStartedEvent(AgendaProgressEvent):
    """Event: A topic has started."""
    type: str = field(default=AgendaEventType.TOPIC_STARTED.value)
    item_id: str = ""
    item_index: int = 0
    transcript_ref: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "itemId": self.item_id,
            "itemIndex": self.item_index,
            "transcriptRef": self.transcript_ref,
            "confidence": self.confidence,
        }


@dataclass
class TopicCompletedEvent(AgendaProgressEvent):
    """Event: A topic has completed."""
    type: str = field(default=AgendaEventType.TOPIC_COMPLETED.value)
    item_id: str = ""
    item_index: int = 0
    transcript_ref: Optional[str] = None
    confidence: float = 0.0
    actual_duration: int = 0  # seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "itemId": self.item_id,
            "itemIndex": self.item_index,
            "transcriptRef": self.transcript_ref,
            "confidence": self.confidence,
            "actualDuration": self.actual_duration,
        }


@dataclass
class TopicSkippedEvent(AgendaProgressEvent):
    """Event: A topic was skipped."""
    type: str = field(default=AgendaEventType.TOPIC_SKIPPED.value)
    item_id: str = ""
    item_index: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "itemId": self.item_id,
            "itemIndex": self.item_index,
            "reason": self.reason,
        }


@dataclass
class AgendaSyncEvent(AgendaProgressEvent):
    """Event: Full agenda sync for late joiners."""
    type: str = field(default=AgendaEventType.AGENDA_SYNC.value)
    agenda: Optional[Dict[str, Any]] = None
    current_item_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "agenda": self.agenda,
            "currentItemIndex": self.current_item_index,
        }


# ============================================================================
# Participant Attribute Types (Late Joiner Sync)
# ============================================================================

@dataclass
class AgendaStateAttribute:
    """
    Compact agenda state stored in agent participant attributes.
    Used for late joiner sync without relying on text stream replay.

    Fields are abbreviated to minimize size:
    - v: version
    - c: current item ID
    - d: done (completed) item IDs
    - s: started timestamp (Unix seconds)
    """
    v: int = 0                # version
    c: Optional[str] = None   # current item ID
    d: List[str] = field(default_factory=list)  # completed item IDs
    s: Optional[int] = None   # meeting started timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "v": self.v,
            "c": self.c,
            "d": self.d,
            "s": self.s,
        }


# ============================================================================
# Topic Detection Types
# ============================================================================

@dataclass
class TopicDetectionResult:
    """Result from LLM topic analysis."""
    has_transitioned: bool = False
    next_topic_id: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class StabilityState:
    """
    Tracks stability for topic detection to prevent thrashing.

    Implements hysteresis: requires K consecutive predictions or
    T seconds of consistent prediction before committing to a switch.
    """
    last_predicted_topic: Optional[str] = None
    consecutive_predictions: int = 0
    first_prediction_time: float = 0.0
    last_switch_time: float = 0.0


# ============================================================================
# Constants
# ============================================================================

# LiveKit topic for agenda events
AGENDA_TOPIC = "hedwiq.agenda"

# Agent identity prefix (must match frontend constant)
AGENT_IDENTITY_PREFIX = "hedwiq"

# ============================================================================
# NEW ARCHITECTURE: Trust the LLM
# ============================================================================
#
# Philosophy: Modern LLMs (GPT-4, Claude, etc.) are intelligent enough to
# understand conversation context. Instead of adding "magic constants" that
# second-guess the LLM, we give it full conversation history and trust its
# judgment.
#
# Old approach (REMOVED):
#   - STABILITY_CONSECUTIVE_K: Required N consecutive predictions
#   - STABILITY_TIME_THRESHOLD: Required N seconds of consistent prediction
#   - SWITCH_CONFIDENCE_THRESHOLD: Arbitrary confidence cutoff
#   - HYSTERESIS_COOLDOWN: Forced delay between switches
#   - MIN_TIME_ON_TOPIC: Minimum time before allowing transition
#
# New approach:
#   - Give LLM full conversation transcript (not just recent segments)
#   - Ask LLM to make intelligent transition decisions
#   - Trust LLM's response without artificial constraints
#   - No confidence thresholds, no stability checks, no cooldowns
#
# Future: Implement context compaction when transcript exceeds limit
# ============================================================================

# Analysis intervals (minimal, just for performance)
MIN_ANALYSIS_INTERVAL = 1.5       # Minimum seconds between analysis runs (for rate limiting only)
ANALYSIS_DEBOUNCE_SECONDS = 1.0   # Debounce delay before analysis (to batch transcript segments)

# Transcript buffer - store FULL conversation for context
# This is a soft limit; we give LLM as much context as possible
MAX_TRANSCRIPT_BUFFER = 100       # Maximum transcript entries to buffer (increased significantly)

# Pre-filter: only skip very short utterances (like "um", "uh")
MIN_SEGMENT_WORDS_FOR_DETECTION = 3  # Minimum words to trigger analysis
