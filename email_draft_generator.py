"""
Email Draft Generator for Luframe Agent - Phase 3 of Real-Time Actions

Generates AI-powered email drafts from classified email-type actions.
Receives actions from ActionClassifier and produces ready-to-send drafts.

Publishing:
- Publishes to luframe.email_draft topic for frontend consumption

Integration:
- ActionClassifier calls on_email_action() when it publishes email-type actions
- EmailDraftGenerator maintains transcript buffer for context
- Uses meeting context (title, participants, agenda) for better drafts
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
    ClassifiedAction,
    EMAIL_ACTION_TYPES,
)
from schemas.email_draft import (
    EmailDraft,
    EmailRecipient,
    MeetingContext,
    DraftStatus,
    DRAFT_GENERATION_TIMEOUT_SECONDS,
    DRAFT_GENERATION_MAX_RETRIES,
    MIN_DRAFT_CONFIDENCE,
    MAX_TRANSCRIPT_CONTEXT_TURNS,
    MAX_TRANSCRIPT_CONTEXT_CHARS,
    MAX_CONCURRENT_GENERATIONS,
    GENERATION_DEBOUNCE_SECONDS,
)
from prompts.email_draft_generation import format_email_draft_prompt
from insight_analyzer import TranscriptEntry
from llm_utils import clean_llm_json_response
from usage_reporter import get_usage_reporter

# Shared identity utilities (for billing attribution)
from utils.identity import (
    extract_user_id_from_identity,
    is_agent_identity,
    get_meeting_owner_from_room,
)

logger = logging.getLogger("luframe-email-draft")

# LiveKit topic for email drafts
EMAIL_DRAFT_TOPIC = "luframe.email_draft"

# Buffer sizes
MAX_CONTEXT_BUFFER = 20        # Number of recent transcript entries for context
PUBLISHED_DRAFT_IDS_MAX = 500  # Deduplication buffer size


@dataclass
class EmailDraftGenerator:
    """
    Generates AI-powered email drafts from email-type actions.

    Receives email-type ClassifiedActions from ActionClassifier and:
    1. Builds context from meeting info and transcript
    2. Generates professional email draft using LLM
    3. Publishes draft to luframe.email_draft topic

    Integration:
    - ActionClassifier calls on_email_action() when it publishes email actions
    - EmailDraftGenerator maintains its own transcript buffer for context
    - Meeting context (title, participants) is set via set_meeting_context()

    Publishing:
    - Topic: luframe.email_draft
    - Attributes: action_id, action_type, status

    Resource Management:
    - Call shutdown() when done to cancel pending tasks
    - Uses bounded deques to prevent memory leaks
    """

    room: rtc.Room
    room_id: str = ""
    llm: Optional[Any] = None

    # Meeting context (set by LuframeAgent when available)
    meeting_context: MeetingContext = field(default_factory=MeetingContext)

    # Transcript context buffer (bounded deque)
    transcript_buffer: Deque[TranscriptEntry] = field(
        default_factory=lambda: deque(maxlen=MAX_CONTEXT_BUFFER)
    )

    # Pending actions queue for debouncing
    pending_actions: List[ClassifiedAction] = field(default_factory=list)

    # Generated drafts for deduplication (bounded deque)
    generated_draft_ids: Deque[str] = field(
        default_factory=lambda: deque(maxlen=PUBLISHED_DRAFT_IDS_MAX)
    )

    # Locks and scheduling
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduled_task: Optional[asyncio.Task] = None

    # Shutdown flag
    _shutdown: bool = False

    # Statistics
    total_generated: int = 0
    generation_errors: int = 0
    low_confidence_skips: int = 0

    def __post_init__(self):
        self.generation_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()
        # Re-initialize deques in case dataclass default_factory didn't work
        if not isinstance(self.transcript_buffer, deque):
            self.transcript_buffer = deque(maxlen=MAX_CONTEXT_BUFFER)
        if not isinstance(self.generated_draft_ids, deque):
            self.generated_draft_ids = deque(maxlen=PUBLISHED_DRAFT_IDS_MAX)

        # Initialize LLM if not provided
        if self.llm is None:
            self.llm = self._create_llm()

        # Initialize meeting context with room_id
        if not isinstance(self.meeting_context, MeetingContext):
            self.meeting_context = MeetingContext(room_id=self.room_id)
        elif not self.meeting_context.room_id and self.room_id:
            # Ensure room_id is set even if meeting_context was provided
            self.meeting_context.room_id = self.room_id

    def _create_llm(self):
        """Create Azure OpenAI LLM client."""
        from livekit.plugins.openai import LLM as OpenAILLM

        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION", "2024-10-01-preview")

        if not azure_endpoint or not api_key:
            logger.warning("Azure OpenAI not configured - email draft generation disabled")
            return None

        return OpenAILLM.with_azure(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def set_meeting_context(
        self,
        title: Optional[str] = None,
        date: Optional[str] = None,
        participants: Optional[List[str]] = None,
        agenda_topics: Optional[List[str]] = None,
        room_id: Optional[str] = None,
    ):
        """
        Set meeting context for better draft generation.

        Called by LuframeAgent when meeting info is available (e.g., from
        agenda tracker or meeting API).

        Args:
            title: Meeting title
            date: Meeting date (ISO or human readable)
            participants: List of participant names
            agenda_topics: List of agenda topics
            room_id: Room identifier for meeting links
        """
        self.meeting_context = MeetingContext(
            meeting_title=title,
            meeting_date=date,
            participants=participants or [],
            agenda_topics=agenda_topics or [],
            room_id=room_id or self.room_id,
        )
        logger.info(f"Meeting context set: {title}, {len(participants or [])} participants")

    def update_participants(self, participants: List[str]):
        """Update participant list (called when participants join/leave)."""
        self.meeting_context.participants = participants

    def update_agenda_topics(self, topics: List[str]):
        """Update agenda topics (called when agenda is loaded/updated)."""
        self.meeting_context.agenda_topics = topics

    async def shutdown(self):
        """
        Gracefully shutdown the generator.

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
            f"EmailDraftGenerator shutdown - generated: {self.total_generated}, "
            f"errors: {self.generation_errors}, skipped: {self.low_confidence_skips}"
        )

    async def add_transcript(self, entry: TranscriptEntry):
        """
        Add transcript entry to context buffer.

        Called by ParticipantTranscriber to maintain context for
        email draft generation.
        """
        if not entry.is_final or self._shutdown:
            return

        # Deque with maxlen handles automatic eviction
        self.transcript_buffer.append(entry)

        # Update participants list from transcript if not already present
        # Use explicit None check since empty string is a valid (though unusual) name
        speaker_name = entry.speaker_name
        if speaker_name is not None and speaker_name.strip() and speaker_name not in self.meeting_context.participants:
            self.meeting_context.participants.append(speaker_name)

    async def on_email_action(self, action: ClassifiedAction):
        """
        Handle a new email-type action from ActionClassifier.

        This is the main entry point - called when ActionClassifier
        publishes an email-type action (email_followup, email_share, email_schedule).

        Args:
            action: The ClassifiedAction with email type
        """
        action_type_str = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        logger.info(
            f"EmailDraftGenerator received email action: [{action_type_str}] "
            f"{action.content[:60]}... (id: {action.id[:8]}...)"
        )

        if self._shutdown:
            logger.warning("EmailDraftGenerator is shutdown, ignoring email action")
            return

        # Validate it's an email action
        if action.action_type not in EMAIL_ACTION_TYPES:
            logger.warning(f"Received non-email action: {action.action_type}")
            return

        # Skip if already generated
        if action.id in self.generated_draft_ids:
            logger.debug(f"Draft already generated for action: {action.id}")
            return

        logger.info(f"EmailDraftGenerator queuing email draft generation for: {action.id[:8]}...")

        # Queue for generation (all checks inside lock)
        async with self.schedule_lock:
            # Backpressure check
            if len(self.pending_actions) >= MAX_CONCURRENT_GENERATIONS:
                logger.warning("Pending generations at limit, dropping action")
                return

            self.pending_actions.append(action)

            # Schedule generation if not already scheduled
            if self.scheduled_task is None or self.scheduled_task.done():
                self.scheduled_task = asyncio.create_task(
                    self._delayed_generation()
                )

    async def _delayed_generation(self):
        """Debounced generation to batch nearby actions."""
        await asyncio.sleep(GENERATION_DEBOUNCE_SECONDS)
        # Check shutdown flag after sleep to avoid unnecessary work
        if self._shutdown:
            return
        await self._run_generation()

    async def _run_generation(self):
        """Run draft generation on pending actions."""
        async with self.generation_lock:
            # Get pending actions
            async with self.schedule_lock:
                if not self.pending_actions:
                    return
                actions_to_generate = self.pending_actions.copy()
                self.pending_actions.clear()

            # Generate draft for each action
            for action in actions_to_generate:
                try:
                    draft = await self._generate_draft(action)
                    if draft:
                        await self._publish_draft(draft)
                        self.total_generated += 1
                except Exception as e:
                    logger.error(f"Draft generation failed for {action.id}: {e}")
                    self.generation_errors += 1

    async def _generate_draft(self, action: ClassifiedAction) -> Optional[EmailDraft]:
        """
        Generate an email draft for a single action.

        Args:
            action: The email-type ClassifiedAction

        Returns:
            EmailDraft if successful, None otherwise
        """
        if self.llm is None:
            logger.warning("LLM not configured - cannot generate draft")
            return None

        # Build transcript context
        transcript_context = self._build_transcript_context()

        # Get metadata hints
        metadata = action.metadata
        recipient_hint = metadata.recipient_hint if metadata else None
        subject_hint = metadata.subject_hint if metadata else None
        urgency = metadata.urgency.value if metadata and hasattr(metadata.urgency, 'value') else "normal"
        datetime_hint = metadata.datetime_hint if metadata else None
        duration_hint = metadata.duration_hint if metadata else None

        # Format prompt
        system_prompt, user_prompt = format_email_draft_prompt(
            action_content=action.content,
            action_type=action.action_type.value if hasattr(action.action_type, 'value') else action.action_type,
            speaker_name=action.speaker_name or "Unknown",
            recipient_hint=recipient_hint,
            subject_hint=subject_hint,
            urgency=urgency,
            meeting_title=self.meeting_context.meeting_title,
            participants=self.meeting_context.participants,
            agenda_topics=self.meeting_context.agenda_topics,
            transcript_context=transcript_context,
            datetime_hint=datetime_hint,
            duration_hint=duration_hint,
        )

        # Call LLM with retry
        for attempt in range(DRAFT_GENERATION_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self._call_llm_generation(system_prompt, user_prompt),
                    timeout=DRAFT_GENERATION_TIMEOUT_SECONDS,
                )

                if result:
                    draft = self._build_email_draft(action, result, transcript_context)
                    if draft:
                        return draft

                # Retry with stricter prompt if parsing failed
                if attempt == 0:
                    user_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No other text."
                    continue

            except asyncio.TimeoutError:
                logger.warning(f"Draft generation timeout (attempt {attempt + 1})")
                if attempt < DRAFT_GENERATION_MAX_RETRIES:
                    continue

            except Exception as e:
                logger.error(f"Draft generation error (attempt {attempt + 1}): {e}")
                if attempt < DRAFT_GENERATION_MAX_RETRIES:
                    continue

        # Failed to generate
        logger.warning(f"Draft generation failed for action: {action.id}")
        return None

    async def _call_llm_generation(
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
        return self._parse_generation_response(response_text)

    def _parse_generation_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into draft result."""
        try:
            # Clean markdown code fences if present
            cleaned = clean_llm_json_response(response)

            if not cleaned:
                return None

            data = json.loads(cleaned)

            # Validate required fields
            if "subject" not in data or "body" not in data:
                logger.warning("Missing required fields in draft response")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse draft JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return None

    def _build_email_draft(
        self,
        action: ClassifiedAction,
        generation_result: Dict,
        transcript_context: str,
    ) -> Optional[EmailDraft]:
        """Build EmailDraft from generation result."""
        try:
            # Extract confidence
            confidence = float(generation_result.get("confidence", 0.8))

            # Skip low confidence drafts
            if confidence < MIN_DRAFT_CONFIDENCE:
                logger.info(f"Skipping low confidence draft ({confidence:.2f}): {action.id}")
                self.low_confidence_skips += 1
                return None

            # Parse recipients
            recipients_data = generation_result.get("recipients", [])
            recipients = []
            for r in recipients_data:
                if isinstance(r, dict) and "name" in r:
                    recipients.append(EmailRecipient(
                        name=r["name"],
                        email=r.get("email"),
                        source=r.get("source", "inferred"),
                    ))

            # Build email body with greeting and closing
            greeting = generation_result.get("greeting", "Hi,")
            body_content = generation_result.get("body", "")
            closing = generation_result.get("closing", "Best regards,")

            # Combine into full body
            full_body = f"{greeting}\n\n{body_content}\n\n{closing}"

            # Get action type value
            action_type_value = (
                action.action_type.value
                if hasattr(action.action_type, 'value')
                else action.action_type
            )

            # Create draft
            draft = EmailDraft(
                action_id=action.id,
                original_insight_id=action.original_insight_id,
                suggested_to=recipients,
                subject=generation_result.get("subject", "Follow-up"),
                body=full_body,
                meeting_context=self.meeting_context,
                transcript_context=transcript_context[:MAX_TRANSCRIPT_CONTEXT_CHARS] if transcript_context else None,
                action_content=action.content,
                action_type=action_type_value,
                speaker_name=action.speaker_name,
                status=DraftStatus.READY,
                generation_confidence=confidence,
            )

            return draft

        except Exception as e:
            logger.error(f"Failed to build EmailDraft: {e}")
            return None

    def _build_transcript_context(self) -> str:
        """Build transcript context from buffer."""
        if not self.transcript_buffer:
            return "No additional context available."

        lines = []
        char_count = 0

        # Get last N turns, respecting character limit
        entries = list(self.transcript_buffer)[-MAX_TRANSCRIPT_CONTEXT_TURNS:]

        for entry in entries:
            # Handle case where both speaker_name and speaker_identity could be None
            speaker = entry.speaker_name or entry.speaker_identity or "Unknown"
            line = f"[{speaker}]: {entry.text}"

            if char_count + len(line) > MAX_TRANSCRIPT_CONTEXT_CHARS:
                break

            lines.append(line)
            char_count += len(line) + 1  # +1 for newline

        return "\n".join(lines) if lines else "No additional context available."

    async def _publish_draft(self, draft: EmailDraft):
        """Publish email draft to LiveKit topic."""
        try:
            # Mark as generated (deque handles eviction)
            self.generated_draft_ids.append(draft.action_id)

            # Build payload
            draft_data = draft.to_dict()

            # Get room_id for attributes
            room_id = draft.meeting_context.room_id or self.room_id or ""

            # Publish to LiveKit
            await self.room.local_participant.send_text(
                json.dumps(draft_data),
                topic=EMAIL_DRAFT_TOPIC,
                attributes={
                    "draft_id": draft.id,
                    "action_id": draft.action_id,
                    "action_type": draft.action_type,
                    "status": draft.status.value if hasattr(draft.status, 'value') else draft.status,
                    "room_id": room_id,
                    "meeting_id": room_id,  # Use roomId as meetingId fallback
                },
            )

            logger.info(
                f"Published email draft: [{draft.action_type}] "
                f"Subject: {draft.subject[:50]}... (confidence: {draft.generation_confidence:.2f})"
            )

            # Report email draft usage to Polar for billing
            # M1 Fix: This awaits (not fire-and-forget) to ensure usage is reported
            # before the function returns, preventing dropped events on serverless
            await self._report_draft_usage(draft, room_id)

        except Exception as e:
            logger.error(f"Failed to publish draft: {e}")

    async def _report_draft_usage(self, draft: EmailDraft, room_id: str):
        """
        Report email draft usage to Polar for billing.

        L4 Fix: Added timeout to prevent slow usage API from delaying subsequent
        draft generations. Draft is already published at this point.

        Args:
            draft: The published email draft
            room_id: The room/meeting ID
        """
        # L4 Fix: 3 second timeout to prevent blocking subsequent generations
        USAGE_REPORT_TIMEOUT = 3.0

        try:
            # Get user_id for usage attribution
            user_id = self._get_meeting_owner_id()

            if not user_id:
                logger.warning(
                    "[EmailDraftGenerator] Could not determine user_id for usage tracking - "
                    "email draft usage will not be billed"
                )
                return

            # Report usage via the singleton UsageReporter with timeout
            reporter = get_usage_reporter()
            try:
                result = await asyncio.wait_for(
                    reporter.report_email_draft(
                        user_id=user_id,
                        count=1,
                        meeting_id=room_id,
                        action_type=draft.action_type,
                    ),
                    timeout=USAGE_REPORT_TIMEOUT,
                )

                if result.success:
                    logger.info(
                        f"[EmailDraftGenerator] Reported email draft usage for user {user_id}"
                    )
                else:
                    logger.warning(
                        f"[EmailDraftGenerator] Failed to report email draft usage: {result.error}"
                    )
            except asyncio.TimeoutError:
                # Log timeout but don't fail - draft is already published
                logger.warning(
                    f"[EmailDraftGenerator] Usage reporting timeout ({USAGE_REPORT_TIMEOUT}s) - "
                    "draft published but billing may be delayed"
                )

        except Exception as e:
            # Don't fail the publish operation on usage reporting errors
            logger.error(f"[EmailDraftGenerator] Error reporting draft usage: {e}")

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return {
            "total_generated": self.total_generated,
            "generation_errors": self.generation_errors,
            "low_confidence_skips": self.low_confidence_skips,
            "pending_count": len(self.pending_actions),
            "buffer_size": len(self.transcript_buffer),
            "meeting_participants": len(self.meeting_context.participants),
        }

    def _get_meeting_owner_id(self) -> Optional[str]:
        """
        Get the meeting owner's user_id for usage tracking.

        H4 Fix: Uses shared identity module (utils.identity) to avoid code duplication.
        C3 Fix: Properly handles UUIDs with hyphens via regex-based suffix extraction.

        Returns:
            User ID of the meeting owner, or None if not determinable
        """
        # Use shared utility for proper UUID extraction
        return get_meeting_owner_from_room(self.room)
