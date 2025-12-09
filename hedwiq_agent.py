"""
Hedwiq Agent - Real-time Transcription and Insight Extraction

A LiveKit agent that provides:
1. Real-time transcription for all meeting participants
2. AI-powered insight extraction using Azure OpenAI
3. Publishing insights via LiveKit text streams

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
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import stt, JobContext, WorkerOptions, cli, AutoSubscribe, llm
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins import silero

from schemas.insights import Insight, InsightType
from prompts.insight_extraction import (
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    INSIGHT_EXTRACTION_USER_TEMPLATE,
)

# Document reference imports (Phase 3: retrieval + LLM alignment)
from document_referencer import DocumentReferencer
from persistent_store import PersistentDocumentStore
from hybrid_retriever import RoomRetrieverManager

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hedwiq-agent")

# Constants
TRANSCRIPTION_TOPIC = "lk.transcription"
INSIGHT_TOPIC = "hedwiq.insight"

# Analysis scheduling constants
MIN_ANALYSIS_INTERVAL = 5.0  # Minimum seconds between analyses
MIN_SEGMENTS_FOR_ANALYSIS = 3  # Minimum new segments before analyzing
ANALYSIS_DELAY = 3.0  # Seconds to wait after last segment before analyzing

# Quality thresholds
MAX_TRANSCRIPT_BUFFER = 30  # Maximum transcript entries to keep for analysis
MIN_CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to publish an insight (raised from 0.6)
MIN_INSIGHT_WORDS = 8  # Minimum words for an insight to be valid
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Word overlap threshold for duplicate detection


@dataclass
class TranscriptEntry:
    """Represents a single transcript entry."""

    speaker_identity: str
    speaker_name: str
    text: str
    timestamp: float
    segment_id: str
    is_final: bool


@dataclass
class InsightAnalyzer:
    """
    Analyzes transcripts and extracts insights using Azure OpenAI.

    This class buffers transcript entries and periodically analyzes them
    for insights, which are then published via text streams.

    Improvements:
    - Queue-based scheduling (no more canceling in-flight LLM calls)
    - Deterministic fingerprint + semantic similarity deduplication
    - Speaker identity mapping for proper transcript linking
    - Previous insights context to avoid repetition
    - Minimum content length filter
    - Retry logic for parsing failures
    """

    room: rtc.Room
    llm: OpenAILLM
    transcript_buffer: List[TranscriptEntry] = field(default_factory=list)
    pending_segments: List[TranscriptEntry] = field(default_factory=list)
    published_insights: set = field(default_factory=set)  # Deterministic fingerprints
    published_contents: List[str] = field(default_factory=list)  # For semantic similarity
    recent_insight_summaries: List[dict] = field(default_factory=list)  # For prompt context
    last_analysis_time: float = 0
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduled_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        self.analysis_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()

    async def add_transcript(self, entry: TranscriptEntry):
        """Add a transcript entry and schedule analysis."""
        # Only process final transcripts
        if not entry.is_final:
            return

        async with self.schedule_lock:
            self.pending_segments.append(entry)

            # Check if we should schedule analysis
            now = time.time()
            time_since_last = now - self.last_analysis_time
            enough_segments = len(self.pending_segments) >= MIN_SEGMENTS_FOR_ANALYSIS
            enough_time = time_since_last >= MIN_ANALYSIS_INTERVAL

            should_schedule = enough_segments or (self.pending_segments and enough_time)

            if should_schedule and (self.scheduled_task is None or self.scheduled_task.done()):
                self.scheduled_task = asyncio.create_task(self._delayed_analysis())

    async def _delayed_analysis(self):
        """Wait for speech pause, then run analysis."""
        await asyncio.sleep(ANALYSIS_DELAY)
        await self._run_analysis()

    async def _run_analysis(self):
        """Run insight extraction with proper locking (max concurrency = 1)."""
        async with self.analysis_lock:
            async with self.schedule_lock:
                if not self.pending_segments:
                    return

                # Move pending to buffer
                segments_to_analyze = self.pending_segments.copy()
                self.pending_segments.clear()

            # Add to main buffer
            self.transcript_buffer.extend(segments_to_analyze)
            if len(self.transcript_buffer) > MAX_TRANSCRIPT_BUFFER:
                self.transcript_buffer = self.transcript_buffer[-MAX_TRANSCRIPT_BUFFER:]

            self.last_analysis_time = time.time()
            await self._extract_insights()

    async def _extract_insights(self):
        """Send buffered transcript to LLM for insight extraction with retry."""
        if not self.transcript_buffer:
            return

        # Build context with speaker identity mapping
        transcript_text, speaker_map = self._build_transcript_context()
        previous_insights = self._build_previous_insights_summary()

        if not transcript_text.strip():
            return

        # Format the prompt with all context
        user_prompt = INSIGHT_EXTRACTION_USER_TEMPLATE.format(
            transcript=transcript_text,
            speaker_map=json.dumps(speaker_map, indent=2),
            previous_insights=previous_insights,
        )

        # Try up to 2 times (retry once on parse failure)
        for attempt in range(2):
            try:
                # Create chat context
                chat_ctx = llm.ChatContext()
                chat_ctx.add_message(role="system", content=INSIGHT_EXTRACTION_SYSTEM_PROMPT)
                chat_ctx.add_message(role="user", content=user_prompt)

                # Call LLM
                response_text = ""
                stream = self.llm.chat(chat_ctx=chat_ctx)
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_text += chunk.delta.content

                # Parse and publish insights
                insights = self._parse_insights(response_text, speaker_map)

                if insights is None and attempt == 0:
                    # Parse failed, retry with stricter prompt
                    logger.warning("JSON parse failed, retrying with stricter prompt")
                    user_prompt += "\n\nIMPORTANT: Return ONLY a valid JSON array. No other text."
                    continue

                if insights:
                    for insight in insights:
                        await self._publish_insight(insight)
                return

            except Exception as e:
                logger.error(f"Insight extraction failed (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    continue
                break

    def _build_transcript_context(self) -> tuple[str, dict]:
        """
        Build transcript context with speaker identity mapping.

        Returns:
            tuple: (transcript_text, speaker_map)
            - transcript_text uses speaker identities for attribution
            - speaker_map maps identity -> display name
        """
        # Merge adjacent same-speaker turns for cleaner context
        merged = self._merge_speaker_turns(self.transcript_buffer[-15:])

        lines = []
        speaker_map = {}

        for turn in merged:
            speaker_map[turn["identity"]] = turn["name"]
            lines.append(f"[{turn['identity']}]: {turn['text']}")

        return "\n".join(lines), speaker_map

    def _merge_speaker_turns(self, entries: List[TranscriptEntry]) -> List[dict]:
        """Merge consecutive entries from the same speaker."""
        if not entries:
            return []

        merged = []
        for entry in entries:
            if merged and merged[-1]["identity"] == entry.speaker_identity:
                # Same speaker, merge text
                merged[-1]["text"] += " " + entry.text
                merged[-1]["segment_id"] = entry.segment_id  # Use latest segment
            else:
                merged.append({
                    "identity": entry.speaker_identity,
                    "name": entry.speaker_name,
                    "text": entry.text,
                    "segment_id": entry.segment_id,
                })
        return merged

    def _build_previous_insights_summary(self) -> str:
        """Build summary of previously extracted insights for context."""
        if not self.recent_insight_summaries:
            return "None yet."

        lines = []
        for insight in self.recent_insight_summaries[-10:]:
            lines.append(f"- [{insight['type']}] {insight['content'][:60]}...")
        return "\n".join(lines)

    def _content_fingerprint(self, insight_type: str, content: str, speaker: str) -> str:
        """Create deterministic fingerprint for deduplication."""
        # Normalize content for fingerprinting
        normalized = f"{insight_type}:{content.lower().strip()}:{speaker or 'unknown'}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _is_semantically_similar(self, new_content: str) -> bool:
        """Check if new insight is too similar to existing ones."""
        new_words = set(new_content.lower().split())

        for existing in self.published_contents[-50:]:  # Check last 50
            existing_words = set(existing.lower().split())
            if not new_words or not existing_words:
                continue

            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)

            if union > 0 and (intersection / union) > SEMANTIC_SIMILARITY_THRESHOLD:
                return True
        return False

    def _parse_insights(self, response: str, speaker_map: dict) -> Optional[List[Insight]]:
        """
        Parse LLM response into Insight objects with improved validation.

        Returns None if parsing fails completely (triggers retry).
        Returns empty list if parsing succeeds but no valid insights.
        """
        try:
            # Clean up response - remove any markdown formatting
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Handle empty response
            if not cleaned or cleaned == "[]":
                return []

            # Parse JSON
            data = json.loads(cleaned)

            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data)}")
                return None  # Trigger retry

            insights = []
            for item in data:
                try:
                    insight = self._validate_and_create_insight(item, speaker_map)
                    if insight:
                        insights.append(insight)
                except Exception as e:
                    logger.warning(f"Failed to parse insight item: {e}")
                    continue

            return insights

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse insights JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return None  # Trigger retry

    def _validate_and_create_insight(self, item: dict, speaker_map: dict) -> Optional[Insight]:
        """Validate a single insight item and create Insight object."""
        # Validate insight type
        insight_type = item.get("type", "").lower()
        if insight_type not in [t.value for t in InsightType]:
            logger.debug(f"Invalid insight type: {insight_type}")
            return None

        # Check confidence threshold
        confidence = float(item.get("confidence", 0.8))
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.debug(f"Low confidence ({confidence}), skipping")
            return None

        content = item.get("content", "").strip()

        # Check minimum content length
        word_count = len(content.split())
        if word_count < MIN_INSIGHT_WORDS:
            logger.debug(f"Content too short ({word_count} words), skipping: {content}")
            return None

        # Check deterministic fingerprint
        speaker_from_llm = item.get("speaker", "")
        fingerprint = self._content_fingerprint(insight_type, content, speaker_from_llm)
        if fingerprint in self.published_insights:
            logger.debug(f"Duplicate fingerprint, skipping: {content[:30]}...")
            return None

        # Check semantic similarity
        if self._is_semantically_similar(content):
            logger.debug(f"Semantically similar to existing, skipping: {content[:30]}...")
            return None

        # Resolve speaker identity
        # The LLM should return the identity token, but fall back to name matching
        speaker_identity = speaker_from_llm
        speaker_name = speaker_from_llm

        if speaker_from_llm in speaker_map:
            # LLM returned correct identity
            speaker_name = speaker_map[speaker_from_llm]
        else:
            # Try to find by name (fallback)
            for identity, name in speaker_map.items():
                if name.lower() == speaker_from_llm.lower():
                    speaker_identity = identity
                    speaker_name = name
                    break
            else:
                # Default to most recent speaker if not found
                if self.transcript_buffer:
                    speaker_identity = self.transcript_buffer[-1].speaker_identity
                    speaker_name = self.transcript_buffer[-1].speaker_name

        # Get transcript reference
        transcript_ref = None
        for entry in reversed(self.transcript_buffer):
            if entry.speaker_identity == speaker_identity:
                transcript_ref = entry.segment_id
                break

        # If still no transcript_ref, use the most recent segment
        if transcript_ref is None and self.transcript_buffer:
            transcript_ref = self.transcript_buffer[-1].segment_id

        # Create insight with millisecond timestamp
        insight = Insight(
            type=InsightType(insight_type),
            content=content,
            speaker=speaker_identity,
            speaker_name=speaker_name,
            confidence=confidence,
            transcript_ref=transcript_ref,
            timestamp=int(time.time() * 1000),  # Milliseconds for frontend
        )

        # Track for deduplication
        self.published_insights.add(fingerprint)
        self.published_contents.append(content)

        # Keep published_contents bounded
        if len(self.published_contents) > 100:
            self.published_contents = self.published_contents[-100:]

        # Track for prompt context
        self.recent_insight_summaries.append({
            "type": insight_type,
            "content": content,
        })
        if len(self.recent_insight_summaries) > 20:
            self.recent_insight_summaries = self.recent_insight_summaries[-20:]

        return insight

    async def _publish_insight(self, insight: Insight):
        """Publish an insight via text stream."""
        try:
            insight_data = {
                "id": str(uuid.uuid4()),
                "type": insight.type,
                "content": insight.content,
                "speaker": insight.speaker,
                "speakerName": insight.speaker_name,
                "confidence": insight.confidence,
                "transcriptRef": insight.transcript_ref,
                "timestamp": insight.timestamp,  # Already in milliseconds
            }

            await self.room.local_participant.send_text(
                json.dumps(insight_data),
                topic=INSIGHT_TOPIC,
                attributes={
                    "insight_type": insight.type,
                    "speaker": insight.speaker or "",
                    "confidence": str(insight.confidence),
                },
            )

            logger.info(
                f"Published insight: [{insight.type}] {insight.content[:50]}..."
            )

        except Exception as e:
            logger.error(f"Failed to publish insight: {e}")


class ParticipantTranscriber:
    """Handles transcription for a single participant's audio track.

    Includes transcript aggregation to combine consecutive speech segments
    into complete utterances before publishing.
    """

    # Aggregation settings
    AGGREGATION_DELAY = 2.0  # Seconds to wait before publishing aggregated transcript
    MAX_AGGREGATION_TIME = 10.0  # Maximum time to aggregate before forcing publish

    def __init__(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant,
        track: rtc.RemoteAudioTrack,
        stt_instance: stt.STT,
        insight_analyzer: InsightAnalyzer,
        document_referencer: Optional[DocumentReferencer] = None,
    ):
        self.room = room
        self.participant = participant
        self.track = track
        self.stt = stt_instance
        self.insight_analyzer = insight_analyzer
        self.document_referencer = document_referencer
        self._task: asyncio.Task | None = None
        self._segment_counter = 0

        # Track segment start time for duration calculation
        self._segment_start_time: Optional[float] = None

        # Transcript aggregation state
        self._aggregation_buffer: List[str] = []
        self._aggregation_segment_id: Optional[str] = None
        self._aggregation_start_time: Optional[float] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._aggregation_lock = asyncio.Lock()

    async def start(self):
        """Start transcribing this participant's audio."""
        self._task = asyncio.create_task(self._transcribe_track())

    async def stop(self):
        """Stop transcribing."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _transcribe_track(self):
        """Process audio from the track and publish transcriptions.

        Uses StreamAdapter with VAD for proper turn detection:
        - START_OF_SPEECH: User started speaking
        - END_OF_SPEECH: User stopped speaking (based on VAD silence detection)
        - FINAL_TRANSCRIPT: Complete transcription of the speech segment
        """
        try:
            audio_stream = rtc.AudioStream(self.track)
            stt_stream = self.stt.stream()

            async def process_audio():
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)
                # Signal end of input when audio stream ends
                stt_stream.end_input()

            async def process_transcriptions():
                current_segment_id = None
                async for event in stt_stream:
                    # Handle start of speech - create new segment
                    if event.type == stt.SpeechEventType.START_OF_SPEECH:
                        self._segment_counter += 1
                        current_segment_id = (
                            f"{self.participant.identity}-{self._segment_counter}"
                        )
                        # Track start time for duration calculation
                        self._segment_start_time = time.time()
                        logger.debug(
                            f"Speech started for {self.participant.identity}, "
                            f"segment: {current_segment_id}"
                        )

                    # Handle end of speech - segment boundary detected by VAD
                    elif event.type == stt.SpeechEventType.END_OF_SPEECH:
                        logger.debug(
                            f"Speech ended for {self.participant.identity}, "
                            f"segment: {current_segment_id}"
                        )

                    # Handle final transcript - complete utterance from STT
                    elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        transcript_text = (
                            event.alternatives[0].text if event.alternatives else ""
                        )
                        if transcript_text.strip():
                            # Ensure we have a segment ID
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = (
                                    f"{self.participant.identity}-{self._segment_counter}"
                                )

                            # Calculate segment duration
                            duration_seconds = 2.0  # Default
                            if self._segment_start_time:
                                duration_seconds = time.time() - self._segment_start_time

                            logger.info(
                                f"[{self.participant.name or self.participant.identity}] {transcript_text}"
                            )

                            # Publish transcription
                            await self._publish_transcription(
                                transcript_text,
                                is_final=True,
                                segment_id=current_segment_id,
                            )

                            # Add to insight analyzer
                            entry = TranscriptEntry(
                                speaker_identity=self.participant.identity,
                                speaker_name=self.participant.name
                                or self.participant.identity,
                                text=transcript_text,
                                timestamp=time.time(),
                                segment_id=current_segment_id,
                                is_final=True,
                            )
                            await self.insight_analyzer.add_transcript(entry)

                            # Send to document referencer for reference detection (Phase 3)
                            if self.document_referencer:
                                await self.document_referencer.on_transcript_final(
                                    segment_id=current_segment_id,
                                    transcript=transcript_text,
                                    speaker_identity=self.participant.identity,
                                    duration_seconds=duration_seconds,
                                )

                            # Reset segment state after final transcript
                            current_segment_id = None
                            self._segment_start_time = None

                    # Note: StreamAdapter doesn't emit INTERIM_TRANSCRIPT
                    # because it waits for complete speech segments via VAD
                    elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                        # This should not be called with StreamAdapter, but handle gracefully
                        transcript_text = (
                            event.alternatives[0].text if event.alternatives else ""
                        )
                        if transcript_text.strip():
                            if current_segment_id is None:
                                self._segment_counter += 1
                                current_segment_id = (
                                    f"{self.participant.identity}-{self._segment_counter}"
                                )

                            await self._publish_transcription(
                                transcript_text,
                                is_final=False,
                                segment_id=current_segment_id,
                            )

            await asyncio.gather(process_audio(), process_transcriptions())

        except asyncio.CancelledError:
            logger.info(f"Stopped transcribing {self.participant.identity}")
            raise
        except Exception as e:
            logger.error(f"Error transcribing {self.participant.identity}: {e}")

    async def _publish_transcription(
        self, text: str, is_final: bool, segment_id: str
    ):
        """Publish transcription to all participants via text stream."""
        try:
            await self.room.local_participant.send_text(
                text,
                topic=TRANSCRIPTION_TOPIC,
                attributes={
                    "lk.transcribed_track_id": self.track.sid,
                    "lk.transcription_final": str(is_final).lower(),
                    "lk.segment_id": segment_id,
                    "speaker_identity": self.participant.identity,
                    "speaker_name": self.participant.name or self.participant.identity,
                },
            )
        except Exception as e:
            logger.error(f"Error publishing transcription: {e}")


class HedwiqAgent:
    """
    Main Hedwiq agent that manages transcription, insight extraction,
    and document reference detection.

    This unified agent (Option A) handles STT, LLM analysis, and document
    retrieval in one process, providing lower latency and simpler deployment.

    Components:
    - ParticipantTranscriber: Per-participant STT with VAD
    - InsightAnalyzer: Queue-based LLM insight extraction
    - DocumentReferencer: Real-time document reference detection (Phase 3)
    """

    def __init__(self, room: rtc.Room, room_id: Optional[str] = None):
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
        base_stt = DeepgramSTT(
            model=self._get_stt_model(),
            language=self._get_stt_language(),
            punctuate=True,
            smart_format=True,
            keyterms=self._get_stt_keyterms(),
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
        self.llm = OpenAILLM.with_azure(
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

        # Initialize document reference detection (Phase 3)
        # Uses persistent store and hybrid retrieval
        self.document_store = PersistentDocumentStore(backend="sqlite")
        self.retriever_manager = RoomRetrieverManager.get_instance(self.document_store)
        self.document_referencer = DocumentReferencer(
            room=room,
            room_id=self.room_id,
            document_store=self.document_store,
            retriever_manager=self.retriever_manager,
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

    def _get_stt_model(self) -> str:
        """Get STT model name (Deepgram) from environment."""
        import os
        return os.getenv("STT_MODEL", "nova-3")

    def _get_stt_language(self) -> str:
        """Get STT language code; set to 'multi' for multilingual meetings."""
        import os
        return os.getenv("STT_LANGUAGE", "en-US")

    def _get_stt_keyterms(self) -> Optional[list[str]]:
        """
        Get keyterms for Deepgram Nova-3 (improves proper noun accuracy).

        Accepts comma-separated list in STT_KEYTERMS env var.
        """
        import os
        raw = os.getenv("STT_KEYTERMS", "")
        terms = [t.strip() for t in raw.split(",") if t.strip()]
        return terms or None

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
        """Stop all transcribers and document referencer."""
        # Stop document referencer
        await self.document_referencer.stop()

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
            self.document_referencer,  # Phase 3: Pass document referencer
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


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Note: Not setting agent_name enables automatic dispatch
            # The agent will automatically join every new room
        )
    )
