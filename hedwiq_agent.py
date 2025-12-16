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
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import stt, JobContext, WorkerOptions, cli, AutoSubscribe
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
        # Set up event handlers
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("track_published", self._on_track_published)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

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


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the Hedwiq agent.

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
    logger.info(f"Hedwiq agent starting for room: {ctx.room.name}")

    # Connect to the room with audio subscription
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Connected to room, starting Hedwiq agent")

    # Create and start the agent
    agent = HedwiqAgent(ctx.room)
    await agent.start()

    logger.info(
        "Hedwiq agent is now listening to all participants "
        "and extracting insights"
    )

    # Keep the agent running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("Agent shutting down")
        await agent.stop()


async def request_handler(req):
    """
    Handle job requests - accept all and set identity prefix to 'hedwiq'.

    This ensures the agent's participant identity starts with 'hedwiq',
    which is required for frontend event filtering (AGENT_IDENTITY_PREFIX).

    NOTE: Do NOT use agent_name in WorkerOptions - that enables explicit dispatch
    and the agent will never join rooms automatically.
    """
    await req.accept(
        # Set identity with hedwiq prefix for frontend filtering
        identity=f"hedwiq-{req.id[:8]}",
        name="Hedwiq Agent",
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_handler,
            # NOTE: Do NOT set agent_name here - it disables automatic dispatch!
            # The agent identity prefix is set via request_handler instead.
        )
    )
