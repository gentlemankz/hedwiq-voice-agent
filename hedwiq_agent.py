"""
Hedwiq Agent - Real-time Transcription, Insight Extraction, and Agenda Tracking

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

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hedwiq-agent")

TRANSCRIPTION_TOPIC = "lk.transcription"


class HedwiqAgent:
    """
    Main Hedwiq agent that manages transcription, insight extraction,
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
        self.document_store = document_store or PersistentDocumentStore(backend="sqlite")
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

        # Check and log meeting limits for monitoring (enforcement is in frontend)
        # This runs async in background to not delay agent startup
        asyncio.create_task(self._check_meeting_limits_async())

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
        # M3 Fix: If humans are still present when agent stops (e.g., shutdown, crash),
        # explicitly set the leave time to now for accurate billing.
        # Without this, _last_human_leave_time would be None and _report_meeting_usage
        # would use time.time() anyway, but this makes the intent explicit.
        if self._human_participant_count > 0 and self._last_human_leave_time is None:
            self._last_human_leave_time = time.time()
            logger.info(
                f"[HedwiqAgent] Agent stopping with {self._human_participant_count} human(s) "
                f"still present, setting leave time to {self._last_human_leave_time:.0f}"
            )

        # Report meeting minutes usage to Polar for billing
        await self._report_meeting_usage()

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

    async def _report_meeting_usage(self):
        """
        Report meeting minutes usage to Polar for billing.

        C4 Fix: Bills based on actual human participant presence, not agent lifetime.
        Duration is measured from first human join to last human leave.

        H2 Fix: Duration is clamped to API limits (1-1440 minutes).

        M2 Fix: Handles negative duration from clock skew.
        """
        # No humans ever joined - nothing to bill
        if self._first_human_join_time is None:
            logger.info(
                "[HedwiqAgent] No human participants joined - skipping usage report"
            )
            return

        # Calculate actual presence duration (C4 fix)
        # Use last human leave time if set, otherwise current time
        end_time = self._last_human_leave_time or time.time()

        # M2 fix: Handle potential negative duration from clock skew
        duration_seconds = max(0, end_time - self._first_human_join_time)

        # H2 fix: Clamp duration to API limits (1-1440 minutes = 24 hours)
        # Round to nearest minute, minimum 1, maximum 1440
        duration_minutes = max(1, min(1440, int(duration_seconds / 60 + 0.5)))

        # Get meeting owner for billing attribution
        user_id = self._get_meeting_owner_id()

        if not user_id:
            logger.warning(
                f"[HedwiqAgent] Could not determine meeting owner for room {self.room_id}. "
                "Meeting minutes will not be billed. This may happen if no human "
                "participants were ever in the room."
            )
            return

        # Report usage via UsageReporter
        try:
            reporter = get_usage_reporter()
            result = await reporter.report_meeting_minutes(
                user_id=user_id,
                minutes=duration_minutes,
                room_id=self.room_id,
            )

            if result.success:
                logger.info(
                    f"[HedwiqAgent] Reported {duration_minutes} meeting minutes "
                    f"for user {user_id} in room {self.room_id} "
                    f"(actual presence: {duration_seconds:.0f}s)"
                )
            else:
                logger.warning(
                    f"[HedwiqAgent] Failed to report meeting usage: {result.error}"
                )
        except Exception as e:
            # Don't fail the shutdown on usage reporting errors
            logger.error(f"[HedwiqAgent] Error reporting meeting usage: {e}")

    async def _check_meeting_limits_async(self):
        """
        Check meeting limits asynchronously for monitoring purposes.

        This is a non-blocking check that logs the user's remaining minutes.
        Actual enforcement should be done by the frontend before room creation.
        The agent does NOT block meetings - it only logs for monitoring.

        This runs after a brief delay to allow the first participant to connect.
        """
        from usage_reporter import PARTICIPANT_WAIT_TIMEOUT_SECONDS
        try:
            # Wait a bit for participants to connect so we can identify the owner
            # L1: Use named constant instead of magic number
            await asyncio.sleep(PARTICIPANT_WAIT_TIMEOUT_SECONDS)

            user_id = self._get_meeting_owner_id()
            if not user_id:
                logger.debug(
                    "[HedwiqAgent] No meeting owner identified yet for limit check"
                )
                return

            reporter = get_usage_reporter()
            allowed, status = await reporter.check_meeting_limits(user_id)

            if allowed:
                logger.info(
                    f"[HedwiqAgent] Meeting limits check: user {user_id} has "
                    f"{status.remaining_minutes} minutes remaining "
                    f"(tier: {status.tier}, used: {status.minutes_used}/{status.minutes_limit})"
                )
            else:
                # Log warning but don't block - frontend should have enforced this
                logger.warning(
                    f"[HedwiqAgent] User {user_id} is over meeting limits! "
                    f"Tier: {status.tier}, Used: {status.minutes_used}/{status.minutes_limit}. "
                    f"Reason: {status.reason}. "
                    "Meeting continues (enforcement should be in frontend)."
                )
        except Exception as e:
            # Don't let limit check failures affect the meeting
            logger.debug(f"[HedwiqAgent] Limit check failed (non-critical): {e}")

    def _get_meeting_owner_id(self) -> Optional[str]:
        """
        Get the meeting owner's user_id for usage tracking.

        Uses the shared identity module (utils.identity) for proper UUID extraction.
        The owner is the first human (non-agent) participant and is responsible
        for the meeting minutes billing.

        Returns:
            User ID of the meeting owner, or None if not determinable
        """
        # Use shared utility which handles caching
        owner = get_meeting_owner_from_room(self.room, self._meeting_owner_id)
        if owner and not self._meeting_owner_id:
            self._meeting_owner_id = owner
        return owner

    def _initialize_existing_participants(self):
        """
        Initialize presence tracking for participants already in the room.

        Called during start() to handle the case where agent joins a room
        that already has human participants. Without this, _first_human_join_time
        would stay None and billing would be skipped.
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
                    f"[HedwiqAgent] Existing human participant found, "
                    f"setting join time to {self._first_human_join_time:.0f}"
                )

            # Cache the meeting owner (first human)
            if not self._meeting_owner_id:
                user_id = extract_user_id_from_identity(participant.identity)
                if user_id:
                    self._meeting_owner_id = user_id
                    logger.info(f"[HedwiqAgent] Meeting owner from existing participant: {user_id}")

        if self._human_participant_count > 0:
            logger.info(
                f"[HedwiqAgent] Initialized with {self._human_participant_count} "
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
                f"[HedwiqAgent] First human participant joined at "
                f"{self._first_human_join_time:.0f}"
            )

        # Cache the meeting owner for usage billing (first human participant)
        # We do this early because the owner might leave before the meeting ends
        if not self._meeting_owner_id:
            user_id = extract_user_id_from_identity(participant.identity)
            if user_id:
                self._meeting_owner_id = user_id
                logger.info(f"[HedwiqAgent] Meeting owner identified: {user_id}")

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
                    f"[HedwiqAgent] Last human participant left at "
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
    Handle job requests - accept all and set identity prefix to 'hedwiq'.

    This ensures the agent's participant identity starts with 'hedwiq',
    which is required for frontend event filtering (AGENT_IDENTITY_PREFIX).

    NOTE: The agent is hidden (invisible to participants) but still maintains
    its identity prefix for internal event routing and stream filtering.
    """
    await req.accept(
        # Set identity with hedwiq prefix for frontend filtering
        identity=f"hedwiq-{req.id[:8]}",
        # No visible name needed since agent is hidden from participants
    )


@server.rtc_session(on_request=request_handler)
async def agent_entrypoint(ctx: JobContext):
    """
    Main entrypoint for the Hedwiq agent (decorated version for AgentServer).

    This agent:
    1. Joins the LiveKit room invisibly (as a hidden agent)
    2. Subscribes to all participant audio tracks
    3. Runs STT (Speech-to-Text) on all audio
    4. Publishes transcriptions via LiveKit text streams (lk.transcription topic)
    5. Analyzes transcripts with Azure OpenAI for insights
    6. Publishes insights via text streams (hedwiq.insight topic)
    7. (Phase 3) Detects document references using hybrid retrieval + LLM alignment
    8. Publishes confirmed references via hedwiq.document_reference topic
    9. (Phase 4) Tracks agenda progress and detects topic transitions
    10. Publishes agenda events via hedwiq.agenda topic
    11. (Phase 1 Real-Time Actions) Classifies action items by execution type
    12. Publishes classified actions via hedwiq.action topic
    13. (Phase 3 Real-Time Actions) Generates email drafts from email-type actions
    14. Publishes email drafts via hedwiq.email_draft topic
    """
    logger.info(f"Hedwiq agent (hidden) starting for room: {ctx.room.name}")

    # Connect to the room with audio subscription
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Connected to room, starting Hedwiq agent")

    # Create and start the agent
    agent = HedwiqAgent(ctx.room)
    await agent.start()

    logger.info(
        "Hedwiq agent is now listening to all participants "
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
