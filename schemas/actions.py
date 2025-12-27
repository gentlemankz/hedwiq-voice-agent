"""
Action Schemas for Luframe Agent

Defines the data models for action classification feature (Phase 1 of Real-Time Actions).
Actions are extracted from meeting insights and classified by execution type.

Key Features:
- ActionType enum for classification categories
- ActionMetadata for extracted hints (recipient, subject, urgency)
- ClassifiedAction model for enhanced action_items with classification
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import time
import uuid


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def generate_action_id() -> str:
    """Generate unique action ID."""
    return f"action-{uuid.uuid4().hex[:12]}"


class ActionType(str, Enum):
    """
    Classification of action items by execution type.

    Each type corresponds to a different execution path:
    - email_* types -> Gmail integration
    - task_create -> Task management integration (future)
    - calendar_event -> Calendar integration
    - manual -> No automation, user handles manually
    """

    # Email-related actions
    EMAIL_FOLLOWUP = "email_followup"      # "send email", "follow up with", "email X about"
    EMAIL_SHARE = "email_share"            # "share with", "send to", "forward to"
    EMAIL_SCHEDULE = "email_schedule"      # "schedule meeting with", "set up call"

    # Task management actions
    TASK_CREATE = "task_create"            # "create task", "add to backlog"

    # Calendar actions
    CALENDAR_EVENT = "calendar_event"      # "block time", "schedule", "remind me"

    # Default fallback
    MANUAL = "manual"                      # No automation, requires manual action


# Email action types that trigger draft generation (Phase 3 of Real-Time Actions)
# Defined as frozenset for immutability and O(1) lookup
EMAIL_ACTION_TYPES = frozenset({
    ActionType.EMAIL_FOLLOWUP,
    ActionType.EMAIL_SHARE,
    ActionType.EMAIL_SCHEDULE,
})


class UrgencyLevel(str, Enum):
    """Urgency level for actions."""
    LOW = "low"           # "when you get a chance", "eventually"
    NORMAL = "normal"     # Default urgency
    HIGH = "high"         # "ASAP", "urgent", "by end of day"
    CRITICAL = "critical" # "immediately", "right now"


class ActionMetadata(BaseModel):
    """
    Extracted metadata from action item for execution.

    Fields are optional as not all actions have all metadata.
    The LLM extracts hints from context that help with execution.
    """

    # Email-related hints
    recipient_hint: Optional[str] = Field(
        None,
        description="Potential email recipient (name/role mentioned)",
        max_length=200
    )
    subject_hint: Optional[str] = Field(
        None,
        description="Potential email subject extracted from context",
        max_length=200
    )

    # Task-related hints
    project_hint: Optional[str] = Field(
        None,
        description="Project or category hint for task creation",
        max_length=100
    )
    assignee_hint: Optional[str] = Field(
        None,
        description="Person assigned to the task",
        max_length=100
    )

    # Calendar-related hints
    datetime_hint: Optional[str] = Field(
        None,
        description="Date/time reference for scheduling",
        max_length=100
    )
    duration_hint: Optional[str] = Field(
        None,
        description="Duration hint (e.g., '30 minutes', '1 hour')",
        max_length=50
    )

    # General metadata
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.NORMAL,
        description="Urgency level of the action"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (camelCase for frontend)."""
        # Handle both enum and string values (due to use_enum_values config)
        urgency_value = (
            self.urgency.value
            if hasattr(self.urgency, 'value')
            else self.urgency
        )
        return {
            "recipientHint": self.recipient_hint,
            "subjectHint": self.subject_hint,
            "projectHint": self.project_hint,
            "assigneeHint": self.assignee_hint,
            "datetimeHint": self.datetime_hint,
            "durationHint": self.duration_hint,
            "urgency": urgency_value,
        }

    # Pydantic v2 configuration
    model_config = ConfigDict(use_enum_values=True)


class ClassifiedAction(BaseModel):
    """
    An action item enhanced with classification and metadata.

    This extends the basic action_item insight with:
    - action_type: Classification for execution path
    - metadata: Extracted hints for email/task/calendar creation
    - requires_email: Quick flag for email-related actions
    - original_insight_id: Links back to the source insight

    Published to the luframe.action topic for frontend consumption.
    """

    id: str = Field(default_factory=generate_action_id)

    # Link to original insight
    original_insight_id: str = Field(..., description="ID of the source action_item insight")

    # Original insight data (copied for convenience)
    content: str = Field(..., min_length=10, max_length=500, description="Action description")
    speaker: Optional[str] = Field(None, description="Speaker identity token")
    speaker_name: Optional[str] = Field(None, description="Speaker display name")
    transcript_ref: Optional[str] = Field(None, description="Reference to transcript segment")

    # Classification results
    action_type: ActionType = Field(..., description="Classification of action type")
    classification_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the classification"
    )

    # Extracted metadata
    metadata: ActionMetadata = Field(default_factory=ActionMetadata)

    # Convenience flags
    requires_email: bool = Field(
        default=False,
        description="True if action type is email_*"
    )

    # Status tracking
    status: str = Field(
        default="detected",
        description="Action status: detected, drafting, draft_ready, sent, rejected"
    )

    # Timestamps
    timestamp: int = Field(
        default_factory=get_timestamp_ms,
        description="When the action was detected (ms)"
    )
    classified_at: int = Field(
        default_factory=get_timestamp_ms,
        description="When classification completed (ms)"
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Set requires_email flag based on action_type
        self.requires_email = self.action_type in [
            ActionType.EMAIL_FOLLOWUP,
            ActionType.EMAIL_SHARE,
            ActionType.EMAIL_SCHEDULE,
        ]

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Ensure content has at least 5 words."""
        word_count = len(v.split())
        if word_count < 5:
            raise ValueError(f"Content must have at least 5 words, got {word_count}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Handle both enum and string values (due to use_enum_values config)
        action_type_value = (
            self.action_type.value
            if hasattr(self.action_type, 'value')
            else self.action_type
        )
        return {
            "id": self.id,
            "original_insight_id": self.original_insight_id,
            "content": self.content,
            "speaker": self.speaker,
            "speakerName": self.speaker_name,
            "transcriptRef": self.transcript_ref,
            "actionType": action_type_value,
            "classificationConfidence": self.classification_confidence,
            "metadata": self.metadata.to_dict(),
            "requiresEmail": self.requires_email,
            "status": self.status,
            "timestamp": self.timestamp,
            "classifiedAt": self.classified_at,
        }

    # Pydantic v2 configuration
    model_config = ConfigDict(use_enum_values=True)


# Classification constants
MIN_CLASSIFICATION_CONFIDENCE = 0.7   # Threshold for auto-classification
CLASSIFICATION_TIMEOUT_SECONDS = 3.0  # LLM call timeout
MAX_CONTEXT_TURNS = 5                 # Surrounding transcript context

# Action type trigger patterns (for reference, actual classification uses LLM)
ACTION_TYPE_PATTERNS = {
    ActionType.EMAIL_FOLLOWUP: [
        "send email", "email about", "follow up with", "reach out to",
        "drop a line", "send a note", "write to"
    ],
    ActionType.EMAIL_SHARE: [
        "share with", "send to", "forward to", "pass along",
        "distribute to", "cc everyone"
    ],
    ActionType.EMAIL_SCHEDULE: [
        "schedule meeting", "set up a call", "book a meeting",
        "arrange a session", "schedule time with"
    ],
    ActionType.TASK_CREATE: [
        "create task", "add to backlog", "file a ticket",
        "create a ticket", "add to sprint", "track this"
    ],
    ActionType.CALENDAR_EVENT: [
        "block time", "add to calendar", "schedule", "remind me",
        "set a reminder", "book time"
    ],
}

# Action type icons for frontend display
ACTION_TYPE_ICONS = {
    ActionType.EMAIL_FOLLOWUP: "mail",
    ActionType.EMAIL_SHARE: "share-2",
    ActionType.EMAIL_SCHEDULE: "calendar-plus",
    ActionType.TASK_CREATE: "list-todo",
    ActionType.CALENDAR_EVENT: "calendar",
    ActionType.MANUAL: "hand",
}
