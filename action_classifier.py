"""
Action Classifier for Hedwiq Agent - Phase 1 of Real-Time Actions

Classifies action items from meeting insights by execution type (email, task, calendar).
Receives action_item insights from InsightAnalyzer and enriches them with:
- Action type classification
- Metadata extraction (recipient hints, urgency, etc.)

Publishing:
- Publishes to hedwiq.action topic for frontend consumption
"""

import asyncio
from collections import deque
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Dict, Any, Callable, Awaitable

from livekit import rtc
from livekit.agents import llm

from schemas.actions import (
    ActionType,
    UrgencyLevel,
    ActionMetadata,
    ClassifiedAction,
    EMAIL_ACTION_TYPES,
    MIN_CLASSIFICATION_CONFIDENCE,
    CLASSIFICATION_TIMEOUT_SECONDS,
)
from schemas.insights import Insight, InsightType
from prompts.action_classification import (
    format_classification_prompt,
    CLASSIFICATION_MAX_RETRIES,
)
from insight_analyzer import TranscriptEntry
from utils import clean_llm_json_response

logger = logging.getLogger("hedwiq-action-classifier")

# Type alias for email action callback
EmailActionCallback = Callable[[ClassifiedAction], Awaitable[None]]

# LiveKit topic for classified actions
ACTION_TOPIC = "hedwiq.action"

# Classification constants
MAX_CONTEXT_BUFFER = 10        # Number of recent transcript entries to use as context
MIN_ACTION_WORDS = 5           # Minimum words in action content
CLASSIFICATION_DEBOUNCE = 0.5  # Seconds to wait before classifying (batch nearby actions)
MAX_PENDING_CLASSIFICATIONS = 10  # Backpressure limit
PUBLISHED_IDS_MAX_SIZE = 500   # Maximum size of published IDs deque for deduplication


@dataclass
class ActionClassifier:
    """
    Classifies action items by execution type.

    Receives action_item insights from InsightAnalyzer and:
    1. Builds context from recent transcript
    2. Classifies using LLM
    3. Extracts metadata (recipient, urgency, etc.)
    4. Publishes enriched action to hedwiq.action topic
    5. (Phase 3) Notifies EmailDraftGenerator for email-type actions

    Integration:
    - InsightAnalyzer calls on_action_item() when it publishes an action_item
    - ActionClassifier maintains its own transcript buffer for context
    - EmailDraftGenerator receives email-type actions via callback

    Publishing:
    - Topic: hedwiq.action
    - Attributes: action_type, requires_email, urgency

    Resource Management:
    - Call shutdown() when done to cancel pending tasks
    - Uses bounded deques to prevent memory leaks
    """

    room: rtc.Room
    llm: Optional[Any] = None

    # Transcript context buffer (bounded deque)
    transcript_buffer: Deque[TranscriptEntry] = field(default_factory=lambda: deque(maxlen=MAX_CONTEXT_BUFFER))

    # Pending actions queue for debouncing
    pending_actions: List[tuple] = field(default_factory=list)  # (insight, insight_id)

    # Published actions for deduplication (bounded deque to prevent memory leak)
    published_action_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=PUBLISHED_IDS_MAX_SIZE))

    # Locks and scheduling
    classification_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduled_task: Optional[asyncio.Task] = None

    # Phase 3 (Real-Time Actions): Callback for email draft generation
    email_action_callback: Optional[EmailActionCallback] = None

    # Shutdown flag
    _shutdown: bool = False

    # Statistics
    total_classified: int = 0
    classification_errors: int = 0
    email_actions_sent: int = 0

    def __post_init__(self):
        self.classification_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()
        # Re-initialize deques in case dataclass default_factory didn't work as expected
        if not isinstance(self.transcript_buffer, deque):
            self.transcript_buffer = deque(maxlen=MAX_CONTEXT_BUFFER)
        if not isinstance(self.published_action_ids, deque):
            self.published_action_ids = deque(maxlen=PUBLISHED_IDS_MAX_SIZE)

        # Initialize LLM if not provided
        if self.llm is None:
            self.llm = self._create_llm()

    def _create_llm(self):
        """Create Azure OpenAI LLM client."""
        from livekit.plugins.openai import LLM as OpenAILLM

        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION", "2024-10-01-preview")

        if not azure_endpoint or not api_key:
            logger.warning("Azure OpenAI not configured - action classification disabled")
            return None

        return OpenAILLM.with_azure(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def set_email_action_callback(self, callback: EmailActionCallback):
        """
        Set callback to be invoked when email-type actions are published.

        Used by EmailDraftGenerator to receive email actions for draft generation.

        Args:
            callback: Async function(action: ClassifiedAction) -> None
        """
        self.email_action_callback = callback

    async def shutdown(self):
        """
        Gracefully shutdown the classifier.

        Cancels any pending scheduled tasks and clears state.
        Call this when the room disconnects or agent shuts down.
        """
        self._shutdown = True

        async with self.schedule_lock:
            if self.scheduled_task and not self.scheduled_task.done():
                self.scheduled_task.cancel()
                try:
                    await self.scheduled_task
                except asyncio.CancelledError:
                    pass
                self.scheduled_task = None

        # Clear pending actions
        self.pending_actions.clear()

        logger.info(
            f"ActionClassifier shutdown - classified: {self.total_classified}, "
            f"errors: {self.classification_errors}, email_actions: {self.email_actions_sent}"
        )

    async def add_transcript(self, entry: TranscriptEntry):
        """
        Add transcript entry to context buffer.

        Called by InsightAnalyzer or ParticipantTranscriber to maintain
        context for classification.
        """
        if not entry.is_final or self._shutdown:
            return

        # Deque with maxlen handles automatic eviction of old entries
        self.transcript_buffer.append(entry)

    async def on_action_item(self, insight: Insight, insight_id: str):
        """
        Handle a new action_item insight from InsightAnalyzer.

        This is the main entry point - called when InsightAnalyzer
        publishes an action_item insight.

        Args:
            insight: The action_item Insight object
            insight_id: The UUID assigned to the insight
        """
        if self._shutdown:
            return

        if insight.type != InsightType.ACTION_ITEM:
            logger.warning(f"Received non-action insight: {insight.type}")
            return

        # Skip short content
        word_count = len(insight.content.split())
        if word_count < MIN_ACTION_WORDS:
            logger.debug(f"Action too short ({word_count} words): {insight.content[:50]}")
            return

        # Queue for classification (all checks inside lock to prevent race conditions)
        async with self.schedule_lock:
            # Skip if already classified (check inside lock)
            if insight_id in self.published_action_ids:
                logger.debug(f"Action already classified: {insight_id}")
                return

            # Backpressure check (inside lock)
            if len(self.pending_actions) >= MAX_PENDING_CLASSIFICATIONS:
                logger.warning("Pending classifications at limit, dropping action")
                return

            self.pending_actions.append((insight, insight_id))

            # Schedule classification if not already scheduled (atomically with lock held)
            if self.scheduled_task is None or self.scheduled_task.done():
                self.scheduled_task = asyncio.create_task(
                    self._delayed_classification()
                )

    async def _delayed_classification(self):
        """Debounced classification to batch nearby actions."""
        await asyncio.sleep(CLASSIFICATION_DEBOUNCE)
        # Check shutdown flag after sleep to avoid unnecessary work
        if self._shutdown:
            return
        await self._run_classification()

    async def _run_classification(self):
        """Run classification on pending actions."""
        async with self.classification_lock:
            # Get pending actions
            async with self.schedule_lock:
                if not self.pending_actions:
                    return
                actions_to_classify = self.pending_actions.copy()
                self.pending_actions.clear()

            # Classify each action
            for insight, insight_id in actions_to_classify:
                try:
                    classified = await self._classify_action(insight, insight_id)
                    if classified:
                        await self._publish_action(classified)
                        self.total_classified += 1
                except Exception as e:
                    logger.error(f"Classification failed for {insight_id}: {e}")
                    self.classification_errors += 1

    async def _classify_action(
        self,
        insight: Insight,
        insight_id: str,
    ) -> Optional[ClassifiedAction]:
        """
        Classify a single action using LLM.

        Args:
            insight: The action_item insight to classify
            insight_id: The UUID of the insight

        Returns:
            ClassifiedAction if successful, None otherwise
        """
        if self.llm is None:
            # No LLM configured - create manual action
            return self._create_manual_action(insight, insight_id)

        # Build transcript context
        context = self._build_transcript_context()

        # Format classification prompt
        system_prompt, user_prompt = format_classification_prompt(
            action_content=insight.content,
            speaker_name=insight.speaker_name or insight.speaker or "Unknown",
            transcript_context=context,
        )

        # Call LLM with retry
        for attempt in range(CLASSIFICATION_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self._call_llm_classification(system_prompt, user_prompt),
                    timeout=CLASSIFICATION_TIMEOUT_SECONDS,
                )

                if result:
                    return self._build_classified_action(insight, insight_id, result)

                # Retry with stricter prompt if parsing failed
                if attempt == 0:
                    user_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No other text."
                    continue

            except asyncio.TimeoutError:
                logger.warning(f"Classification timeout (attempt {attempt + 1})")
                if attempt < CLASSIFICATION_MAX_RETRIES:
                    continue

            except Exception as e:
                logger.error(f"Classification error (attempt {attempt + 1}): {e}")
                if attempt < CLASSIFICATION_MAX_RETRIES:
                    continue

        # Fallback to manual classification
        logger.warning(f"Classification failed, defaulting to manual: {insight_id}")
        return self._create_manual_action(insight, insight_id)

    async def _call_llm_classification(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[Dict]:
        """
        Call LLM and parse JSON response.

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=system_prompt)
        chat_ctx.add_message(role="user", content=user_prompt)

        response_text = ""
        stream = self.llm.chat(chat_ctx=chat_ctx)
        try:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    response_text += chunk.delta.content
        finally:
            # Ensure stream resources are released
            if hasattr(stream, 'aclose'):
                await stream.aclose()

        # Parse JSON response
        return self._parse_classification_response(response_text)

    def _parse_classification_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into classification result."""
        try:
            # Clean markdown code fences if present
            cleaned = clean_llm_json_response(response)

            if not cleaned:
                return None

            data = json.loads(cleaned)

            # Validate required fields
            if "action_type" not in data:
                logger.warning("Missing action_type in classification response")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classification JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return None

    def _build_classified_action(
        self,
        insight: Insight,
        insight_id: str,
        classification: Dict,
    ) -> Optional[ClassifiedAction]:
        """Build ClassifiedAction from classification result."""
        try:
            # Parse action type
            action_type_str = classification.get("action_type", "manual").lower()
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning(f"Unknown action type: {action_type_str}")
                action_type = ActionType.MANUAL

            # Parse confidence
            confidence = float(classification.get("confidence", 0.8))

            # If confidence too low, default to manual
            if confidence < MIN_CLASSIFICATION_CONFIDENCE:
                action_type = ActionType.MANUAL

            # Parse metadata
            metadata_dict = classification.get("metadata", {})
            urgency_str = metadata_dict.get("urgency", "normal").lower()
            try:
                urgency = UrgencyLevel(urgency_str)
            except ValueError:
                urgency = UrgencyLevel.NORMAL

            metadata = ActionMetadata(
                recipient_hint=metadata_dict.get("recipient_hint"),
                subject_hint=metadata_dict.get("subject_hint"),
                assignee_hint=metadata_dict.get("assignee_hint"),
                datetime_hint=metadata_dict.get("datetime_hint"),
                duration_hint=metadata_dict.get("duration_hint"),
                urgency=urgency,
            )

            # Create classified action
            return ClassifiedAction(
                original_insight_id=insight_id,
                content=insight.content,
                speaker=insight.speaker,
                speaker_name=insight.speaker_name,
                transcript_ref=insight.transcript_ref,
                action_type=action_type,
                classification_confidence=confidence,
                metadata=metadata,
                timestamp=insight.timestamp,
            )

        except Exception as e:
            logger.error(f"Failed to build ClassifiedAction: {e}")
            return None

    def _create_manual_action(
        self,
        insight: Insight,
        insight_id: str,
    ) -> ClassifiedAction:
        """Create a manual action when classification fails or is unavailable."""
        return ClassifiedAction(
            original_insight_id=insight_id,
            content=insight.content,
            speaker=insight.speaker,
            speaker_name=insight.speaker_name,
            transcript_ref=insight.transcript_ref,
            action_type=ActionType.MANUAL,
            classification_confidence=1.0,  # High confidence it's manual
            metadata=ActionMetadata(),
            timestamp=insight.timestamp,
        )

    def _build_transcript_context(self) -> str:
        """Build transcript context from buffer."""
        if not self.transcript_buffer:
            return "No additional context available."

        lines = []
        # Deque already limited by maxlen, iterate directly
        for entry in self.transcript_buffer:
            speaker = entry.speaker_name or entry.speaker_identity
            lines.append(f"[{speaker}]: {entry.text}")

        return "\n".join(lines)

    async def _publish_action(self, action: ClassifiedAction):
        """Publish classified action to LiveKit topic."""
        try:
            # Mark as published (deque with maxlen handles eviction)
            self.published_action_ids.append(action.original_insight_id)

            # Build payload
            action_data = action.to_dict()

            # Publish to LiveKit
            await self.room.local_participant.send_text(
                json.dumps(action_data),
                topic=ACTION_TOPIC,
                attributes={
                    "action_type": action.action_type.value,
                    "requires_email": str(action.requires_email).lower(),
                    "urgency": action.metadata.urgency.value,
                    "original_insight_id": action.original_insight_id,
                },
            )

            logger.info(
                f"Published action: [{action.action_type.value}] "
                f"{action.content[:50]}... (confidence: {action.classification_confidence:.2f})"
            )

            # Phase 3 (Real-Time Actions): Notify EmailDraftGenerator for email-type actions
            if action.action_type in EMAIL_ACTION_TYPES and self.email_action_callback:
                try:
                    await self.email_action_callback(action)
                    self.email_actions_sent += 1
                except Exception as callback_error:
                    # Don't let callback errors affect action publishing
                    logger.warning(f"Email action callback failed: {callback_error}")

        except Exception as e:
            logger.error(f"Failed to publish action: {e}")

    def get_stats(self) -> Dict:
        """Get classification statistics."""
        return {
            "total_classified": self.total_classified,
            "classification_errors": self.classification_errors,
            "email_actions_sent": self.email_actions_sent,
            "pending_count": len(self.pending_actions),
            "buffer_size": len(self.transcript_buffer),
        }
