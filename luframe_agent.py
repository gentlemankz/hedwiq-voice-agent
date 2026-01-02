"""
Luframe Agent - Real-time Transcription, Insight Extraction, and Agenda Tracking

A LiveKit agent that provides:
1. Real-time transcription for all meeting participants
2. AI-powered insight extraction using Azure OpenAI
3. Publishing insights via LiveKit text streams
4. (Phase 4) Automatic agenda topic detection and progress tracking

This is the main unified agent (Option A from PHASE2_INSIGHTS_PLAN.md).

Improvements implemented:
- Queue-based analysis instead of cancel-based debouncing
- Deterministic fingerprint deduplication + semantic similarity check
- Speaker identity mapping for proper transcript linking
- Timestamp in milliseconds for frontend compatibility
- Previous insights context to avoid repetition
- Minimum content length filter
- Retry logic for LLM parsing failures
- Merged adjacent same-speaker turns

Phase 4 additions:
- AgendaTracker for automatic topic detection
- Multi-signal detection (explicit phrases, keywords, LLM)
- Stability/hysteresis to prevent topic thrashing
- Participant attributes for late joiner sync
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import stt, JobContext, cli, AutoSubscribe, AgentServer, WorkerPermissions
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins import silero

from insight_analyzer import InsightAnalyzer
from participant_transcriber import ParticipantTranscriber

# Document reference imports (Phase 3: retrieval + LLM alignment)
from document_referencer import DocumentReferencer
from persistent_store import PersistentDocumentStore
from hybrid_retriever import RoomRetrieverManager
from transcription_config import get_stt_keyterms, get_stt_language, get_stt_model

# Agenda tracking imports (Phase 4)
from agenda_tracker import AgendaTracker

# Action classification imports (Phase 1 of Real-Time Actions)
from action_classifier import ActionClassifier

# Email draft generation imports (Phase 3 of Real-Time Actions)
from email_draft_generator import EmailDraftGenerator

# Usage tracking imports (Polar billing integration)
from usage_reporter import get_usage_reporter, close_usage_reporter

# Shared identity utilities (for billing attribution)
from utils.identity import (
    extract_user_id_from_identity,
    is_agent_identity,
    get_meeting_owner_from_room,
)
from secret_manager import load_secrets_to_env

# Load secrets from Docker secrets files (production) first, then .env (development)
load_secrets_to_env()  # Loads /run/secrets/* into environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")  # Fills in any missing from .env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luframe-agent")

TRANSCRIPTION_TOPIC = "lk.transcription"
LIMIT_EXCEEDED_TOPIC = "luframe.limit_exceeded"

# Secondary limit enforcement constants
LIMIT_CHECK_INTERVAL_SECONDS = 60  # Re-check limits every 60 seconds
GRACE_PERIOD_SECONDS = 60  # Allow 60 seconds after limit exceeded before stopping
OWNER_VERIFICATION_TIMEOUT_SECONDS = 30  # SECURITY: Max time to wait for owner verification before fail-closed


class LuframeAgent:
    """
    Main Luframe agent that manages transcription, insight extraction,
    document reference detection, agenda tracking, and action classification.

    This unified agent (Option A) handles STT, LLM analysis, document
    retrieval, agenda tracking, and action classification in one process,
    providing lower latency and simpler deployment.

    Components:
    - ParticipantTranscriber: Per-participant STT with VAD
    - InsightAnalyzer: Queue-based LLM insight extraction
    - DocumentReferencer: Real-time document reference detection (Phase 3)
    - AgendaTracker: Automatic topic detection and progress tracking (Phase 4)
    - ActionClassifier: Action item classification for automation (Phase 1 of Real-Time Actions)
    - EmailDraftGenerator: AI-generated email drafts from actions (Phase 3 of Real-Time Actions)
    """

    def __init__(
        self,
        room: rtc.Room,
        room_id: Optional[str] = None,
        *,
        stt_adapter: Optional[stt.STT] = None,
        llm_client: Optional[OpenAILLM] = None,
        document_store: Optional[PersistentDocumentStore] = None,
        retriever_manager: Optional[RoomRetrieverManager] = None,
        document_referencer: Optional[DocumentReferencer] = None,
        agenda_tracker: Optional[AgendaTracker] = None,
        action_classifier: Optional[ActionClassifier] = None,
        email_draft_generator: Optional[EmailDraftGenerator] = None,
    ):
        self.room = room
        self.room_id = room_id or room.name
        self.transcribers: Dict[str, ParticipantTranscriber] = {}

        # Usage tracking for Polar billing - track actual participant presence
        # C4 Fix: Bill based on human participant presence, not agent lifetime
        self._first_human_join_time: Optional[float] = None
        self._last_human_leave_time: Optional[float] = None
        self._human_participant_count: int = 0
        self._meeting_owner_id: Optional[str] = None
        # SECURITY FIX: Verified owner from database (not first-joiner assumption)
        self._verified_meeting_owner_id: Optional[str] = None

        # Periodic usage reporting - report every N minutes to prevent data loss on crash
        # This ensures minutes are captured even if agent is killed without graceful shutdown
        self._periodic_report_task: Optional[asyncio.Task] = None
        self._last_reported_minutes: int = 0  # Track what we've already reported
        self._usage_report_interval_seconds: int = 300  # Report every 5 minutes

        # Secondary limit enforcement - agent-side backup to frontend checks
        self._limit_exceeded: bool = False
        self._limit_exceeded_time: Optional[float] = None
        self._limit_check_task: Optional[asyncio.Task] = None
        self._services_stopped: bool = False
        # SECURITY FIX: Track when we started waiting for owner verification
        self._owner_verification_start_time: Optional[float] = None

        # Initialize VAD (Silero) for proper turn detection
        # This prevents transcription fragmentation by detecting natural speech boundaries
        # For meeting transcription, we use longer silence duration to capture complete thoughts
        self.vad = silero.VAD.load(
            min_speech_duration=0.1,    # Minimum 100ms to start detecting speech
            min_silence_duration=1.2,   # Wait 1.2 seconds of silence before ending speech (meeting-optimized)
            prefix_padding_duration=0.5, # Include 500ms of audio before speech starts
            activation_threshold=0.45,  # Slightly more sensitive to catch soft speech
        )

        # Initialize base STT (Deepgram) with configurable model/language
        if stt_adapter:
            self.stt = stt_adapter
        else:
            base_stt = DeepgramSTT(
                model=get_stt_model(),
                language=get_stt_language(),
                punctuate=True,
                smart_format=True,
                keyterms=get_stt_keyterms(),
            )
            # Wrap STT with VAD using StreamAdapter
            # This buffers audio until VAD detects end of speech, then sends complete
            # segments to Deepgram - preventing word-by-word fragmentation
            self.stt = stt.StreamAdapter(stt=base_stt, vad=self.vad)

        # Initialize LLM (Azure OpenAI)
        # Uses environment variables:
        # - AZURE_OPENAI_API_KEY
        # - AZURE_OPENAI_ENDPOINT
        # - OPENAI_API_VERSION
        # - AZURE_OPENAI_DEPLOYMENT (optional, defaults to gpt-4o-mini)
        self.llm = llm_client or OpenAILLM.with_azure(
            azure_deployment=self._get_azure_deployment(),
            azure_endpoint=self._get_azure_endpoint(),
            api_key=self._get_azure_api_key(),
            api_version=self._get_azure_api_version(),
        )

        # Initialize insight analyzer
        self.insight_analyzer = InsightAnalyzer(
            room=room,
            llm=self.llm,
        )

        # Initialize action classifier (Phase 1 of Real-Time Actions)
        # Classifies action_item insights by execution type (email, task, calendar)
        self.action_classifier = action_classifier or ActionClassifier(
            room=room,
            llm=self.llm,
        )

        # Connect action classifier to insight analyzer
        # When InsightAnalyzer publishes an action_item, it notifies ActionClassifier
        self.insight_analyzer.set_action_item_callback(
            self.action_classifier.on_action_item
        )

        # Initialize email draft generator (Phase 3 of Real-Time Actions)
        # Generates AI-powered email drafts from email-type actions
        self.email_draft_generator = email_draft_generator or EmailDraftGenerator(
            room=room,
            room_id=self.room_id,
            llm=self.llm,
        )

        # Connect email draft generator to action classifier
        # When ActionClassifier publishes an email-type action, it notifies EmailDraftGenerator
        self.action_classifier.set_email_action_callback(
            self.email_draft_generator.on_email_action
        )

        # Initialize document reference detection (Phase 3)
        # Uses persistent store and hybrid retrieval
        # Read backend config from environment (set in docker-compose.yml)
        doc_backend = os.getenv("DOCUMENT_STORE_BACKEND", "sqlite")
        doc_db_path = os.getenv("DB_PATH")
        doc_storage_dir = os.getenv("DOCUMENT_STORAGE_DIR", str(Path(__file__).parent / "document_storage"))
        self.document_store = document_store or PersistentDocumentStore(
            backend=doc_backend,
            db_path=doc_db_path,
            storage_dir=doc_storage_dir,
        )
        self.retriever_manager = retriever_manager or RoomRetrieverManager.get_instance(self.document_store)
        self.document_referencer = document_referencer or DocumentReferencer(
            room=room,
            room_id=self.room_id,
            document_store=self.document_store,
            retriever_manager=self.retriever_manager,
        )

        # Initialize agenda tracking (Phase 4)
        # Tracks meeting progress and detects topic transitions
        self.agenda_tracker = agenda_tracker or AgendaTracker(
            room=room,
            room_id=self.room_id,
            llm=self.llm,
        )

    def _get_azure_deployment(self) -> str:
        """Get Azure OpenAI deployment name from environment."""
        import os
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    def _get_azure_endpoint(self) -> str:
        """Get Azure OpenAI endpoint from environment."""
        import os
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        return endpoint

    def _get_azure_api_key(self) -> str:
        """Get Azure OpenAI API key from environment."""
        import os
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
        return api_key

    def _get_azure_api_version(self) -> str:
        """Get Azure OpenAI API version from environment."""
        import os
        return os.getenv("OPENAI_API_VERSION", "2024-10-01-preview")

    async def start(self):
        """Start the agent - listen for participants and their audio tracks."""
        # Set up event handlers FIRST before processing existing participants
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("track_published", self._on_track_published)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        # Initialize presence tracking for participants already in the room
        # This handles the case where agent joins a room with existing humans
        self._initialize_existing_participants()

        # SECURITY FIX: Fetch verified meeting owner from database
        # This prevents billing the wrong person if an attacker joins first
        asyncio.create_task(self._fetch_verified_meeting_owner())

        # Check meeting limits (initial check + secondary enforcement)
        # This runs async in background to not delay agent startup
        asyncio.create_task(self._check_meeting_limits_async())

        # Start periodic limit checking for secondary enforcement
        # This catches users who exceed limits during a meeting
        self._limit_check_task = asyncio.create_task(self._periodic_limit_check())

        # Start periodic usage reporting to prevent data loss on crash
        # Reports incremental minutes every 5 minutes during the meeting
        self._periodic_report_task = asyncio.create_task(self._periodic_usage_reporter())

        # Start document referencer (Phase 3)
        await self.document_referencer.start()

        # Start agenda tracker (Phase 4)
        # This is designed to fail gracefully - if it fails, transcription continues
        try:
            await self.agenda_tracker.start()
        except Exception as e:
            logger.error(f"Failed to start agenda tracker: {e}. Agenda tracking disabled.")

        logger.info(
            f"Found {len(self.room.remote_participants)} remote participants"
        )

        # Subscribe to existing participants' audio tracks
        for participant in self.room.remote_participants.values():
            logger.info(
                f"Checking participant: {participant.identity}, "
                f"tracks: {len(participant.track_publications)}"
            )
            for track_pub in participant.track_publications.values():
                logger.info(
                    f"  Track: {track_pub.sid}, kind: {track_pub.kind}, "
                    f"subscribed: {track_pub.subscribed}"
                )
                if (
                    track_pub.track
                    and track_pub.kind == rtc.TrackKind.KIND_AUDIO
                    and isinstance(track_pub.track, rtc.RemoteAudioTrack)
                ):
                    await self._start_transcriber(participant, track_pub.track)

    async def stop(self):
        """Stop all transcribers, document referencer, agenda tracker, action classifier, and email draft generator."""
        # Cancel periodic usage reporter first
        if self._periodic_report_task and not self._periodic_report_task.done():
            self._periodic_report_task.cancel()
            try:
                await self._periodic_report_task
            except asyncio.CancelledError:
                pass
            logger.info("[LuframeAgent] Periodic usage reporter stopped")

        # Cancel periodic limit checker
        if self._limit_check_task and not self._limit_check_task.done():
            self._limit_check_task.cancel()
            try:
                await self._limit_check_task
            except asyncio.CancelledError:
                pass
            logger.info("[LuframeAgent] Periodic limit checker stopped")

        # M3 Fix: If humans are still present when agent stops (e.g., shutdown, crash),
        # explicitly set the leave time to now for accurate billing.
        # Without this, _last_human_leave_time would be None and _report_meeting_usage
        # would use time.time() anyway, but this makes the intent explicit.
        if self._human_participant_count > 0 and self._last_human_leave_time is None:
            self._last_human_leave_time = time.time()
            logger.info(
                f"[LuframeAgent] Agent stopping with {self._human_participant_count} human(s) "
                f"still present, setting leave time to {self._last_human_leave_time:.0f}"
            )

        # Report remaining meeting minutes usage to Polar for billing
        # This reports only minutes not yet reported by periodic reporter
        await self._report_meeting_usage(final=True)

        # Stop document referencer
        await self.document_referencer.stop()

        # Stop agenda tracker (Phase 4)
        await self.agenda_tracker.stop()

        # Stop action classifier (Phase 1 - Real-Time Actions)
        await self.action_classifier.shutdown()

        # Stop email draft generator (Phase 3 - Real-Time Actions)
        await self.email_draft_generator.shutdown()

        # Stop all transcribers
        for transcriber in self.transcribers.values():
            await transcriber.stop()
        self.transcribers.clear()

    async def _periodic_usage_reporter(self):
        """
        Periodically report meeting minutes during the meeting.

        This ensures that even if the agent crashes or is killed without graceful
        shutdown, most of the meeting minutes will have been reported.

        Reports incremental minutes (delta since last report) every N minutes.
        """
        logger.info(
            f"[LuframeAgent] Periodic usage reporter started "
            f"(interval: {self._usage_report_interval_seconds}s)"
        )

        try:
            while True:
                await asyncio.sleep(self._usage_report_interval_seconds)

                # Only report if humans are/were in the meeting
                if self._first_human_join_time is None:
                    continue

                # Calculate current total minutes
                end_time = self._last_human_leave_time or time.time()
                duration_seconds = max(0, end_time - self._first_human_join_time)
                total_minutes = int(duration_seconds / 60)

                # Calculate delta (minutes not yet reported)
                minutes_to_report = total_minutes - self._last_reported_minutes

                if minutes_to_report > 0:
                    await self._report_meeting_usage(
                        final=False,
                        minutes_override=minutes_to_report
                    )
                    self._last_reported_minutes = total_minutes
                    logger.info(
                        f"[LuframeAgent] Periodic report: {minutes_to_report} minutes "
                        f"(total reported so far: {self._last_reported_minutes})"
                    )

        except asyncio.CancelledError:
            logger.info("[LuframeAgent] Periodic usage reporter cancelled")
            raise

    async def _report_meeting_usage(self, final: bool = False, minutes_override: Optional[int] = None):
        """
        Report meeting minutes usage to Polar for billing.

        Args:
            final: If True, this is the final report at meeting end. Reports only
                   unreported minutes (total - already_reported).
            minutes_override: If provided, report this exact number of minutes
                              (used by periodic reporter).

        C4 Fix: Bills based on actual human participant presence, not agent lifetime.
        Duration is measured from first human join to last human leave.

        H2 Fix: Duration is clamped to API limits (1-1440 minutes).

        M2 Fix: Handles negative duration from clock skew.

        Incremental Reporting Fix: Reports delta minutes during meeting to prevent
        data loss on crash.
        """
        # Debug: Log entry into this function
        logger.debug(
            f"[LuframeAgent] _report_meeting_usage called: "
            f"final={final}, minutes_override={minutes_override}, "
            f"first_join={self._first_human_join_time}, "
            f"last_leave={self._last_human_leave_time}, "
            f"human_count={self._human_participant_count}, "
            f"owner_id={self._meeting_owner_id}"
        )

        # No humans ever joined - nothing to bill
        if self._first_human_join_time is None:
            logger.info(
                "[LuframeAgent] No human participants joined - skipping usage report"
            )
            return

        # Calculate actual presence duration (C4 fix)
        # Use last human leave time if set, otherwise current time
        end_time = self._last_human_leave_time or time.time()

        # M2 fix: Handle potential negative duration from clock skew
        duration_seconds = max(0, end_time - self._first_human_join_time)

        # H2 fix: Clamp duration to API limits (1-1440 minutes = 24 hours)
        # Round to nearest minute, minimum 1, maximum 1440
        total_duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        logger.debug(
            f"[LuframeAgent] Duration calculation: "
            f"start={self._first_human_join_time:.0f}, end={end_time:.0f}, "
            f"duration_seconds={duration_seconds:.0f}, total_minutes={total_duration_minutes}, "
            f"already_reported={self._last_reported_minutes}"
        )

        # Determine minutes to report
        if minutes_override is not None:
            # Explicit override from periodic reporter
            minutes_to_report = minutes_override
        elif final:
            # Final report: only report unreported minutes
            minutes_to_report = total_duration_minutes - self._last_reported_minutes
            if minutes_to_report <= 0:
                logger.info(
                    f"[LuframeAgent] Final report: all {total_duration_minutes} minutes "
                    f"already reported via periodic reports"
                )
                return
        else:
            # Legacy behavior: report full duration (should not happen with new code)
            minutes_to_report = total_duration_minutes

        # Get meeting owner for billing attribution
        user_id = self._get_meeting_owner_id()

        if not user_id:
            logger.warning(
                f"[LuframeAgent] Could not determine meeting owner for room {self.room_id}. "
                "Meeting minutes will not be billed. This may happen if no human "
                "participants were ever in the room."
            )
            return

        logger.info(
            f"[LuframeAgent] Reporting {minutes_to_report} minutes for user {user_id} "
            f"(room={self.room_id}, final={final}, source=agent)"
        )

        # Report usage via UsageReporter (with retry built-in)
        try:
            reporter = get_usage_reporter()
            result = await reporter.report_meeting_minutes(
                user_id=user_id,
                minutes=minutes_to_report,
                room_id=self.room_id,
                session_id=self.room_id,  # Use room_id as session identifier for idempotency dedup
                source="agent",  # Identify this as agent-reported for deduplication
            )

            if result.success:
                if final:
                    self._last_reported_minutes = total_duration_minutes
                logger.info(
                    f"[LuframeAgent] SUCCESS: {'Final' if final else 'Incremental'} report - "
                    f"{minutes_to_report} minutes for user {user_id} in room {self.room_id} "
                    f"(total presence: {duration_seconds:.0f}s, total reported: {self._last_reported_minutes})"
                )
            else:
                logger.error(
                    f"[LuframeAgent] FAILED to report meeting usage after retries: {result.error} "
                    f"(user={user_id}, minutes={minutes_to_report}, room={self.room_id})"
                )
        except Exception as e:
            # Don't fail the shutdown on usage reporting errors
            logger.error(
                f"[LuframeAgent] EXCEPTION during meeting usage report: {type(e).__name__}: {e} "
                f"(user={user_id}, minutes={minutes_to_report}, room={self.room_id})"
            )

    async def _check_meeting_limits_async(self, initial_check: bool = True):
        """
        Check meeting limits and enforce them as secondary protection.

        This is a SECONDARY enforcement mechanism - the frontend should check
        limits before allowing meeting creation. However, if limits are exceeded
        (e.g., free user bypasses frontend check, or limit reached during meeting),
        this agent-side check will:
        1. Publish a limit_exceeded event to notify the frontend
        2. Stop premium agent services (transcription, insights, etc.)

        The meeting room itself stays open (users can still talk) but without
        AI features. This provides a graceful degradation experience.

        Args:
            initial_check: If True, waits for participants to connect first.
        """
        import json
        from usage_reporter import PARTICIPANT_WAIT_TIMEOUT_SECONDS

        try:
            if initial_check:
                # Wait a bit for participants to connect so we can identify the owner
                await asyncio.sleep(PARTICIPANT_WAIT_TIMEOUT_SECONDS)

            user_id = self._get_meeting_owner_id()
            if not user_id:
                # SECURITY FIX: Fail CLOSED when owner cannot be verified
                # Start tracking when we first encountered this
                if self._owner_verification_start_time is None:
                    self._owner_verification_start_time = time.time()
                    logger.warning(
                        f"[LuframeAgent] SECURITY: No meeting owner verified for room {self.room_id}. "
                        f"Starting {OWNER_VERIFICATION_TIMEOUT_SECONDS}s grace period before fail-closed."
                    )
                    return

                # Check if grace period has expired
                elapsed = time.time() - self._owner_verification_start_time
                if elapsed >= OWNER_VERIFICATION_TIMEOUT_SECONDS:
                    logger.error(
                        f"[LuframeAgent] SECURITY FAIL-CLOSED: Could not verify owner for room {self.room_id} "
                        f"after {elapsed:.0f}s. Stopping premium services as security precaution. "
                        "This may be: (1) Meeting created without DB record, (2) DB lookup failure, "
                        "(3) Malicious attempt to bypass billing."
                    )

                    # Publish a specific limit_exceeded event for unknown owner
                    await self._publish_limit_exceeded_event(
                        user_id="unknown",
                        tier="free",
                        minutes_used=0,
                        minutes_limit=0,
                        reason="Unable to verify meeting owner - premium services stopped",
                    )

                    # Stop premium services immediately
                    if not self._services_stopped:
                        await self._stop_premium_services()
                else:
                    logger.debug(
                        f"[LuframeAgent] Waiting for owner verification ({elapsed:.0f}s / "
                        f"{OWNER_VERIFICATION_TIMEOUT_SECONDS}s)"
                    )
                return

            reporter = get_usage_reporter()
            allowed, status = await reporter.check_meeting_limits(user_id)

            # Owner was successfully verified - reset verification timer
            if self._owner_verification_start_time is not None:
                logger.info(f"[LuframeAgent] Owner verified for room {self.room_id}: {user_id}")
                self._owner_verification_start_time = None

            if allowed:
                logger.info(
                    f"[LuframeAgent] Meeting limits check: user {user_id} has "
                    f"{status.remaining_minutes} minutes remaining "
                    f"(tier: {status.tier}, used: {status.minutes_used}/{status.minutes_limit})"
                )
                # If services were stopped due to limits, restart them (user upgraded)
                if self._services_stopped:
                    logger.info(
                        f"[LuframeAgent] User {user_id} now has available minutes - "
                        "restarting premium services (likely upgraded subscription)"
                    )
                    await self._restart_premium_services()
                # Reset limit exceeded state if previously exceeded
                self._limit_exceeded = False
                self._limit_exceeded_time = None
            else:
                # SECONDARY ENFORCEMENT: Limits exceeded
                logger.warning(
                    f"[LuframeAgent] LIMIT ENFORCEMENT: User {user_id} is over meeting limits! "
                    f"Tier: {status.tier}, Used: {status.minutes_used}/{status.minutes_limit}. "
                    f"Reason: {status.reason}"
                )

                # Track when limit was first exceeded
                if not self._limit_exceeded:
                    self._limit_exceeded = True
                    self._limit_exceeded_time = time.time()

                # Publish limit exceeded event to frontend
                await self._publish_limit_exceeded_event(
                    user_id=user_id,
                    tier=status.tier,
                    minutes_used=status.minutes_used,
                    minutes_limit=status.minutes_limit,
                    reason=status.reason,
                )

                # After grace period, stop premium services
                if self._limit_exceeded_time:
                    elapsed = time.time() - self._limit_exceeded_time
                    if elapsed >= GRACE_PERIOD_SECONDS and not self._services_stopped:
                        logger.warning(
                            f"[LuframeAgent] Grace period expired ({elapsed:.0f}s), "
                            "stopping premium services for user over limits"
                        )
                        await self._stop_premium_services()

        except Exception as e:
            # Don't let limit check failures affect the meeting
            logger.error(f"[LuframeAgent] Limit check failed: {e}")

    async def _publish_limit_exceeded_event(
        self,
        user_id: str,
        tier: str,
        minutes_used: int,
        minutes_limit: int,
        reason: Optional[str] = None,
    ):
        """
        Publish a limit exceeded event to the frontend via LiveKit text stream.

        The frontend should listen to the 'luframe.limit_exceeded' topic and
        display an upgrade prompt to the user.
        """
        import json

        try:
            event_data = {
                "type": "limit_exceeded",
                "user_id": user_id,
                "tier": tier,
                "minutes_used": minutes_used,
                "minutes_limit": minutes_limit,
                "reason": reason or "Monthly meeting minutes limit reached",
                "grace_period_seconds": GRACE_PERIOD_SECONDS,
                "timestamp": int(time.time() * 1000),  # ms for frontend
            }

            await self.room.local_participant.send_text(
                json.dumps(event_data),
                topic=LIMIT_EXCEEDED_TOPIC,
                attributes={
                    "event_type": "limit_exceeded",
                    "tier": tier,
                    "user_id": user_id,
                },
            )

            logger.info(
                f"[LuframeAgent] Published limit_exceeded event: "
                f"user={user_id}, tier={tier}, used={minutes_used}/{minutes_limit}"
            )
        except Exception as e:
            logger.error(f"[LuframeAgent] Failed to publish limit_exceeded event: {e}")

    async def _stop_premium_services(self):
        """
        Stop premium agent services when user exceeds limits.

        This disables transcription, insights, document references, and other
        AI-powered features. The meeting room stays open (users can talk)
        but without AI assistance.

        This is a graceful degradation - better than abruptly ending the meeting.
        """
        if self._services_stopped:
            return

        self._services_stopped = True
        logger.warning("[LuframeAgent] Stopping premium services due to limit exceeded")

        try:
            # Stop all transcribers
            for key, transcriber in list(self.transcribers.items()):
                try:
                    await transcriber.stop()
                    logger.debug(f"[LuframeAgent] Stopped transcriber: {key}")
                except Exception as e:
                    logger.debug(f"[LuframeAgent] Error stopping transcriber {key}: {e}")
            self.transcribers.clear()

            # Stop insight analyzer (it will stop processing)
            # The analyzer doesn't have a stop method, but clearing transcribers
            # prevents new segments from being added

            # Stop document referencer
            if self.document_referencer:
                await self.document_referencer.stop()

            # Stop agenda tracker
            if self.agenda_tracker:
                await self.agenda_tracker.stop()

            # Stop action classifier
            if self.action_classifier:
                await self.action_classifier.shutdown()

            # Stop email draft generator
            if self.email_draft_generator:
                await self.email_draft_generator.shutdown()

            # Publish final notification
            import json
            await self.room.local_participant.send_text(
                json.dumps({
                    "type": "services_stopped",
                    "reason": "limit_exceeded",
                    "message": "Meeting AI features have been disabled due to usage limits. Upgrade your plan to restore features.",
                    "timestamp": int(time.time() * 1000),
                }),
                topic=LIMIT_EXCEEDED_TOPIC,
                attributes={"event_type": "services_stopped"},
            )

            logger.warning(
                "[LuframeAgent] Premium services stopped. "
                "Meeting continues without AI features."
            )
        except Exception as e:
            logger.error(f"[LuframeAgent] Error stopping premium services: {e}")

    async def _restart_premium_services(self):
        """
        Restart premium agent services after user upgrades their subscription.

        This re-enables transcription, insights, document references, and other
        AI-powered features that were previously disabled due to limit exceeded.

        Called when periodic limit check detects user now has available minutes
        (e.g., after upgrading their subscription mid-meeting).
        """
        if not self._services_stopped:
            return

        logger.info("[LuframeAgent] Restarting premium services after subscription upgrade")

        try:
            # Reset state
            self._services_stopped = False
            self._limit_exceeded = False
            self._limit_exceeded_time = None

            # Restart document referencer
            if self.document_referencer:
                await self.document_referencer.start()
                logger.debug("[LuframeAgent] Document referencer restarted")

            # Restart agenda tracker
            if self.agenda_tracker:
                try:
                    await self.agenda_tracker.start()
                    logger.debug("[LuframeAgent] Agenda tracker restarted")
                except Exception as e:
                    logger.error(f"Failed to restart agenda tracker: {e}")

            # Action classifier and email draft generator will be re-connected
            # when insights come in (they don't need explicit restart)

            # Restart transcribers for all current participants with audio tracks
            for participant in self.room.remote_participants.values():
                for track_pub in participant.track_publications.values():
                    if (
                        track_pub.track
                        and track_pub.kind == rtc.TrackKind.KIND_AUDIO
                        and isinstance(track_pub.track, rtc.RemoteAudioTrack)
                    ):
                        await self._start_transcriber(participant, track_pub.track)
                        logger.debug(f"[LuframeAgent] Restarted transcriber for {participant.identity}")

            # Publish service restored notification
            import json
            await self.room.local_participant.send_text(
                json.dumps({
                    "type": "services_restored",
                    "reason": "subscription_upgraded",
                    "message": "Meeting AI features have been restored. Thank you for upgrading!",
                    "timestamp": int(time.time() * 1000),
                }),
                topic=LIMIT_EXCEEDED_TOPIC,
                attributes={"event_type": "services_restored"},
            )

            logger.info(
                "[LuframeAgent] Premium services RESTARTED successfully. "
                "AI features are now available again."
            )
        except Exception as e:
            logger.error(f"[LuframeAgent] Error restarting premium services: {e}")
            # If restart fails, keep services stopped to avoid inconsistent state
            self._services_stopped = True

    async def _periodic_limit_check(self):
        """
        Periodically check limits during the meeting.

        This catches cases where:
        - User started within limits but exceeded during the meeting
        - Frontend check was bypassed somehow
        - User UPGRADED mid-meeting (services should be restarted)

        IMPORTANT: Continue checking even after services are stopped.
        This allows detecting subscription upgrades and restarting services.
        """
        try:
            while True:
                await asyncio.sleep(LIMIT_CHECK_INTERVAL_SECONDS)

                # Run limit check (handles both stopping and restarting services)
                await self._check_meeting_limits_async(initial_check=False)
        except asyncio.CancelledError:
            logger.debug("[LuframeAgent] Periodic limit check cancelled")
        except Exception as e:
            logger.error(f"[LuframeAgent] Periodic limit check error: {e}")

    def _get_meeting_owner_id(self) -> Optional[str]:
        """
        Get the meeting owner's user_id for usage tracking.

        SECURITY FIX #7: Now ONLY uses verified owner from database.
        The first-joiner fallback has been removed to prevent billing attacks
        where an attacker joins first to become the billing target.

        If the database lookup fails, we return None and usage is NOT billed.
        This is fail-closed behavior - better to miss billing than bill the wrong person.

        Returns:
            User ID of the meeting owner, or None if not determinable
        """
        # ONLY use verified owner from database - no fallbacks
        if self._verified_meeting_owner_id:
            return self._verified_meeting_owner_id

        # SECURITY FIX #7: Do NOT fall back to first-joiner
        # If we don't have a verified owner, log error and return None
        # The meeting will not be billed, but this is safer than billing the wrong person
        logger.error(
            f"[LuframeAgent] SECURITY: No verified owner for room {self.room_id}. "
            "Meeting usage will NOT be billed to prevent billing attacks. "
            "Ensure meetings are created through the frontend API."
        )
        return None

    async def _fetch_verified_meeting_owner(self) -> Optional[str]:
        """
        Fetch the verified meeting owner from the database.

        SECURITY FIX: This queries the actual meeting host from the database
        rather than assuming the first joiner is the host.

        Returns:
            Verified host user ID, or None if not determinable
        """
        try:
            reporter = get_usage_reporter()
            verified_owner = await reporter.get_meeting_host(self.room_id)

            if verified_owner:
                self._verified_meeting_owner_id = verified_owner
                self._meeting_owner_id = verified_owner  # Also update legacy cache
                logger.info(f"[LuframeAgent] Verified meeting owner from DB: {verified_owner}")
                return verified_owner
            else:
                # SECURITY FIX #7: Do NOT fall back to first-joiner logic
                # If owner cannot be verified, usage will NOT be billed (fail-closed)
                logger.warning(
                    f"[LuframeAgent] Could not verify meeting owner for room {self.room_id}. "
                    "Meeting usage will NOT be billed to prevent billing attacks. "
                    "Ensure meetings are created through the frontend API."
                )
                return None
        except Exception as e:
            logger.error(f"[LuframeAgent] Error fetching verified owner: {e}")
            return None

    def _initialize_existing_participants(self):
        """
        Initialize presence tracking for participants already in the room.

        Called during start() to handle the case where agent joins a room
        that already has human participants. Without this, _first_human_join_time
        would stay None and billing would be skipped.

        SECURITY FIX #7: No longer caches meeting owner from first participant.
        Owner is ONLY set from verified database lookup.
        """
        for participant in self.room.remote_participants.values():
            # Skip agent participants
            if is_agent_identity(participant.identity):
                continue

            # Count this human participant
            self._human_participant_count += 1

            # Record first human join time (use current time as best approximation)
            if self._first_human_join_time is None:
                self._first_human_join_time = time.time()
                logger.info(
                    f"[LuframeAgent] Existing human participant found, "
                    f"setting join time to {self._first_human_join_time:.0f}"
                )

            # SECURITY FIX #7: Do NOT cache meeting owner from first participant
            # The owner MUST come from the verified database lookup only
            # This prevents billing attacks where attacker joins first

        if self._human_participant_count > 0:
            logger.info(
                f"[LuframeAgent] Initialized with {self._human_participant_count} "
                f"existing human participant(s)"
            )

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle new audio track subscription."""
        logger.info(
            f"Track subscribed: {track.sid}, kind: {track.kind}, "
            f"from: {participant.identity}"
        )
        if track.kind == rtc.TrackKind.KIND_AUDIO and isinstance(
            track, rtc.RemoteAudioTrack
        ):
            logger.info(
                f"Starting transcription for audio track from {participant.identity}"
            )
            asyncio.create_task(self._start_transcriber(participant, track))

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle audio track unsubscription."""
        key = f"{participant.identity}:{track.sid}"
        if key in self.transcribers:
            asyncio.create_task(self.transcribers[key].stop())
            del self.transcribers[key]

    def _on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle when a track is published."""
        logger.info(
            f"Track published: {publication.sid}, kind: {publication.kind}, "
            f"from: {participant.identity}"
        )

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle when a new participant joins."""
        logger.info(
            f"Participant connected: {participant.identity}, "
            f"name: {participant.name}"
        )

        # Skip agent participants for billing tracking
        if is_agent_identity(participant.identity):
            return

        # C4 Fix: Track human participant count for presence-based billing
        self._human_participant_count += 1

        # Record first human join time for billing
        if self._first_human_join_time is None:
            self._first_human_join_time = time.time()
            logger.info(
                f"[LuframeAgent] First human participant joined at "
                f"{self._first_human_join_time:.0f}"
            )

        # SECURITY FIX #7: Do NOT cache meeting owner from first participant
        # The owner MUST come from the verified database lookup only (_fetch_verified_meeting_owner)
        # This prevents billing attacks where an attacker joins first to become the billing target
        # The legacy _meeting_owner_id field is only set from _fetch_verified_meeting_owner now

        # Phase 4: Notify agenda tracker for late joiner sync
        # This publishes agenda sync event so late joiners get current state
        if self.agenda_tracker:
            asyncio.create_task(
                self._safe_on_participant_connected(participant.identity)
            )

    async def _safe_on_participant_connected(self, participant_identity: str):
        """Safely call agenda tracker on_participant_connected with error handling."""
        try:
            await self.agenda_tracker.on_participant_connected(participant_identity)
        except Exception as e:
            logger.warning(f"Failed to sync agenda for late joiner: {e}")

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnection."""
        # C4 Fix: Track human participant disconnection for presence-based billing
        if not is_agent_identity(participant.identity):
            self._human_participant_count = max(0, self._human_participant_count - 1)

            # Record when room becomes empty of humans
            if self._human_participant_count == 0:
                self._last_human_leave_time = time.time()
                logger.info(
                    f"[LuframeAgent] Last human participant left at "
                    f"{self._last_human_leave_time:.0f}"
                )

        # Clean up transcribers for this participant
        keys_to_remove = [
            k for k in self.transcribers if k.startswith(f"{participant.identity}:")
        ]
        for key in keys_to_remove:
            asyncio.create_task(self.transcribers[key].stop())
            del self.transcribers[key]

    async def _start_transcriber(
        self, participant: rtc.RemoteParticipant, track: rtc.RemoteAudioTrack
    ):
        """Start a transcriber for a participant's audio track."""
        key = f"{participant.identity}:{track.sid}"
        if key in self.transcribers:
            return  # Already transcribing this track

        logger.info(
            f"Starting transcription for {participant.name or participant.identity}"
        )
        transcriber = ParticipantTranscriber(
            self.room,
            participant,
            track,
            self.stt,
            self.insight_analyzer,
            self.document_referencer,      # Phase 3: Pass document referencer
            self.agenda_tracker,           # Phase 4: Pass agenda tracker
            self.action_classifier,        # Phase 1 (Real-Time Actions): Pass action classifier
            self.email_draft_generator,    # Phase 3 (Real-Time Actions): Pass email draft generator
            transcription_topic=TRANSCRIPTION_TOPIC,
        )
        self.transcribers[key] = transcriber
        await transcriber.start()


# =============================================================================
# Agent Server Configuration - Hidden Agent
# =============================================================================

# Create AgentServer with hidden permissions
# Hidden agent is invisible to other participants while maintaining full functionality
# This provides a "bot-free" user experience while secretly providing AI-powered features
server = AgentServer(
    permissions=WorkerPermissions(
        can_publish=False,       # No video/audio tracks needed
        can_subscribe=True,      # Subscribe to participant audio for transcription
        can_publish_data=True,   # Publish text streams (transcription, insights, etc.)
        hidden=True,             # INVISIBLE to other participants
    ),
)


async def request_handler(req):
    """
    Handle job requests - accept all and set identity prefix to 'luframe'.

    This ensures the agent's participant identity starts with 'luframe',
    which is required for frontend event filtering (AGENT_IDENTITY_PREFIX).

    NOTE: The agent is hidden (invisible to participants) but still maintains
    its identity prefix for internal event routing and stream filtering.
    """
    await req.accept(
        # Set identity with luframe prefix for frontend filtering
        identity=f"luframe-{req.id[:8]}",
        # No visible name needed since agent is hidden from participants
    )


@server.rtc_session(on_request=request_handler)
async def agent_entrypoint(ctx: JobContext):
    """
    Main entrypoint for the Luframe agent (decorated version for AgentServer).

    This agent:
    1. Joins the LiveKit room invisibly (as a hidden agent)
    2. Subscribes to all participant audio tracks
    3. Runs STT (Speech-to-Text) on all audio
    4. Publishes transcriptions via LiveKit text streams (lk.transcription topic)
    5. Analyzes transcripts with Azure OpenAI for insights
    6. Publishes insights via text streams (luframe.insight topic)
    7. (Phase 3) Detects document references using hybrid retrieval + LLM alignment
    8. Publishes confirmed references via luframe.document_reference topic
    9. (Phase 4) Tracks agenda progress and detects topic transitions
    10. Publishes agenda events via luframe.agenda topic
    11. (Phase 1 Real-Time Actions) Classifies action items by execution type
    12. Publishes classified actions via luframe.action topic
    13. (Phase 3 Real-Time Actions) Generates email drafts from email-type actions
    14. Publishes email drafts via luframe.email_draft topic
    """
    logger.info(f"Luframe agent (hidden) starting for room: {ctx.room.name}")

    # Connect to the room with audio subscription
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Connected to room, starting Luframe agent")

    # Create and start the agent
    agent = LuframeAgent(ctx.room)
    await agent.start()

    logger.info(
        "Luframe agent is now listening to all participants "
        "and extracting insights (hidden from participant list)"
    )

    # Keep the agent running
    # L2 Fix: Use try/finally to ensure cleanup on ALL exit paths
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Agent shutting down (cancelled)")
    except Exception as e:
        logger.error(f"Agent error: {e}")
    finally:
        # Always clean up, regardless of how we exit
        await agent.stop()
        # Clean up usage reporter HTTP client
        await close_usage_reporter()
        logger.info("Agent cleanup complete")


if __name__ == "__main__":
    cli.run_app(server)
