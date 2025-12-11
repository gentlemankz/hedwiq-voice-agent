"""
Agenda Tracker for Hedwiq Agent

Tracks meeting progress through agenda items using LLM analysis.
Receives agenda from frontend via LiveKit text stream, analyzes transcripts
to detect topic transitions, and publishes progress updates.

Pipeline:
1. Receive agenda from frontend via LiveKit stream (hedwiq.agenda topic)
2. Accumulate transcript segments as they arrive
3. Periodically analyze if current topic is complete
4. Publish progress updates to frontend (hedwiq.agenda_progress topic)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from livekit import rtc
from livekit.agents import llm

from insight_analyzer import TranscriptEntry
from schemas.agenda import (
    Agenda,
    AgendaItem,
    AgendaItemStatus,
    AgendaProgressType,
    AgendaProgressUpdate,
    TopicAnalysisResult,
    MIN_SEGMENTS_FOR_ANALYSIS,
    MIN_ANALYSIS_INTERVAL_SECONDS,
    ANALYSIS_DELAY_SECONDS,
    MIN_CONFIDENCE_FOR_COMPLETION,
    MAX_TRANSCRIPT_WINDOW,
    MIN_SEGMENTS_SINCE_TOPIC_START,
    TRANSITION_COOLDOWN_SECONDS,
)
from prompts.agenda_tracking import (
    AGENDA_TRACKING_SYSTEM_PROMPT,
    format_tracking_prompt,
)

logger = logging.getLogger("hedwiq-agent")

# LiveKit topics for agenda communication
AGENDA_TOPIC = "hedwiq.agenda"
AGENDA_PROGRESS_TOPIC = "hedwiq.agenda_progress"


@dataclass
class AgendaTracker:
    """
    Tracks meeting progress through agenda items using LLM analysis.

    This component:
    1. Receives agenda from frontend when user joins the room
    2. Accumulates transcript segments for analysis
    3. Uses LLM to detect when topics are completed
    4. Publishes progress updates back to frontend

    Attributes:
        room: LiveKit room instance for publishing updates
        room_id: Identifier for the current room
        llm: Azure OpenAI LLM client for topic analysis
    """

    room: rtc.Room
    room_id: str
    llm: any

    # Agenda state
    agenda: Optional[Agenda] = None
    current_item_index: int = field(default=-1)  # -1 = not started
    item_start_times: Dict[int, float] = field(default_factory=dict)
    item_statuses: Dict[int, AgendaItemStatus] = field(default_factory=dict)

    # Transcript buffer for analysis
    transcript_buffer: List[TranscriptEntry] = field(default_factory=list)
    pending_segments: List[TranscriptEntry] = field(default_factory=list)
    segments_since_topic_start: int = field(default=0)

    # Analysis scheduling
    last_analysis_time: float = field(default=0)
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduled_task: Optional[asyncio.Task] = None

    # Deduplication and metrics
    published_transitions: Set[str] = field(default_factory=set)
    last_transition_time: float = field(default=0)
    total_analyses: int = field(default=0)
    successful_transitions: int = field(default=0)

    # Running state
    _running: bool = field(default=False)
    _stream_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        self.analysis_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()
        self.item_start_times = {}
        self.item_statuses = {}
        self.published_transitions = set()

    async def start(self):
        """Start listening for agenda from frontend via LiveKit text stream."""
        self._running = True

        # Register text stream handler for agenda topic
        # Frontend sends agenda via sendText() which requires register_text_stream_handler()
        # NOT room.on("data_received") which is for data packets
        try:
            self.room.register_text_stream_handler(
                AGENDA_TOPIC,
                self._on_agenda_text_stream
            )
            logger.info(f"AgendaTracker registered text stream handler for topic '{AGENDA_TOPIC}'")
        except ValueError as e:
            # Handler already registered (e.g., reconnection scenario)
            logger.warning(f"Text stream handler already registered: {e}")

        logger.info(f"AgendaTracker started for room {self.room_id}")

    async def stop(self):
        """Stop tracker and log final metrics."""
        self._running = False

        # Unregister text stream handler
        try:
            self.room.unregister_text_stream_handler(AGENDA_TOPIC)
            logger.debug(f"Unregistered text stream handler for topic '{AGENDA_TOPIC}'")
        except Exception:
            # Handler wasn't registered or already unregistered
            pass

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self.scheduled_task and not self.scheduled_task.done():
            self.scheduled_task.cancel()
            try:
                await self.scheduled_task
            except asyncio.CancelledError:
                pass

        # Log metrics
        logger.info(
            f"AgendaTracker stopped for room {self.room_id}. "
            f"Analyses: {self.total_analyses}, "
            f"Transitions: {self.successful_transitions}"
        )

    def _on_agenda_text_stream(self, reader, participant_identity: str):
        """
        Handle incoming text stream for agenda topic.

        This is the correct handler for LiveKit text streams sent via sendText().
        The reader provides async access to the stream content.

        Args:
            reader: TextStreamReader with read_all() method
            participant_identity: Identity of the participant who sent the stream
        """
        # Create async task to read and process the stream
        asyncio.create_task(self._process_agenda_stream(reader, participant_identity))

    async def _process_agenda_stream(self, reader, participant_identity: str):
        """Process the agenda text stream asynchronously."""
        try:
            # Read the full content from the stream
            payload = await reader.read_all()
            logger.debug(f"Received agenda stream from {participant_identity}: {payload[:100]}...")
            await self._handle_agenda_message(payload)
        except Exception as e:
            logger.error(f"Error processing agenda text stream: {e}")

    async def _handle_agenda_message(self, payload: str):
        """Process incoming agenda message from frontend."""
        try:
            data = json.loads(payload)
            message_type = data.get("type")

            if message_type == "agenda_init":
                await self.on_agenda_received(data.get("agenda", {}))
            else:
                logger.debug(f"Unknown agenda message type: {message_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse agenda message: {e}")
        except Exception as e:
            logger.error(f"Error processing agenda message: {e}")

    async def on_agenda_received(self, agenda_data: dict):
        """
        Handle agenda initialization from frontend.

        Called when frontend sends the agenda via hedwiq.agenda topic.
        """
        try:
            # Parse agenda items
            items_data = agenda_data.get("items", [])
            items = []
            for idx, item_data in enumerate(items_data):
                item = AgendaItem(
                    id=item_data.get("id", str(uuid.uuid4())),
                    title=item_data.get("title", f"Topic {idx + 1}"),
                    description=item_data.get("description"),
                    estimated_minutes=item_data.get("estimatedMinutes"),
                    lead_by=item_data.get("leadBy"),
                    order=item_data.get("order", idx),
                )
                items.append(item)

            # Create agenda
            self.agenda = Agenda(
                id=agenda_data.get("id", str(uuid.uuid4())),
                room_id=agenda_data.get("roomId", self.room_id),
                items=items,
            )

            # Initialize statuses
            for idx in range(len(items)):
                self.item_statuses[idx] = AgendaItemStatus.PENDING

            # Reset tracking state
            self.current_item_index = -1
            self.segments_since_topic_start = 0
            self.transcript_buffer.clear()
            self.pending_segments.clear()

            # Reset deduplication and timing state from any prior agenda
            self.published_transitions.clear()
            self.item_start_times.clear()
            self.last_transition_time = 0
            self.last_analysis_time = 0
            self.total_analyses = 0
            self.successful_transitions = 0

            logger.info(
                f"Received agenda '{self.agenda.id}' with {len(items)} items: "
                f"{[item.title for item in items]}"
            )

        except Exception as e:
            logger.error(f"Failed to parse agenda: {e}")

    async def add_transcript(self, entry: TranscriptEntry):
        """
        Add a transcript entry and schedule analysis.

        Called by ParticipantTranscriber for each final transcript segment.
        """
        if not self._running or not entry.is_final:
            return

        # Skip if no agenda loaded
        if not self.agenda:
            logger.debug(f"[AgendaTracker] Skipping transcript - no agenda loaded")
            return

        async with self.schedule_lock:
            self.pending_segments.append(entry)
            self.segments_since_topic_start += 1

            now = time.time()
            time_since_last = now - self.last_analysis_time
            enough_segments = len(self.pending_segments) >= MIN_SEGMENTS_FOR_ANALYSIS
            enough_time = time_since_last >= MIN_ANALYSIS_INTERVAL_SECONDS

            # Log current state for debugging
            logger.debug(
                f"[AgendaTracker] Transcript added - "
                f"pending={len(self.pending_segments)}, "
                f"segments_since_start={self.segments_since_topic_start}, "
                f"current_item={self.current_item_index}, "
                f"time_since_last={time_since_last:.1f}s, "
                f"enough_segments={enough_segments}, "
                f"enough_time={enough_time}"
            )

            # Check if we should analyze
            should_analyze = (
                enough_segments or
                (self.pending_segments and enough_time)
            ) and self.segments_since_topic_start >= MIN_SEGMENTS_SINCE_TOPIC_START

            # Also analyze if agenda hasn't started yet (to detect first topic)
            if self.current_item_index == -1 and len(self.pending_segments) >= 2:
                should_analyze = True
                logger.debug(f"[AgendaTracker] Will analyze for first topic detection")

            if should_analyze:
                if self.scheduled_task is None or self.scheduled_task.done():
                    logger.info(f"[AgendaTracker] Scheduling analysis - segments={len(self.pending_segments)}, segments_since_start={self.segments_since_topic_start}")
                    self.scheduled_task = asyncio.create_task(self._delayed_analysis())
                else:
                    logger.debug(f"[AgendaTracker] Analysis already scheduled, skipping")
            else:
                reasons = []
                if not enough_segments:
                    reasons.append(f"need {MIN_SEGMENTS_FOR_ANALYSIS} pending segments (have {len(self.pending_segments)})")
                if not enough_time:
                    reasons.append(f"need {MIN_ANALYSIS_INTERVAL_SECONDS}s since last analysis (only {time_since_last:.1f}s)")
                if self.segments_since_topic_start < MIN_SEGMENTS_SINCE_TOPIC_START:
                    reasons.append(f"need {MIN_SEGMENTS_SINCE_TOPIC_START} segments since topic start (have {self.segments_since_topic_start})")
                logger.debug(f"[AgendaTracker] Not analyzing yet: {', '.join(reasons)}")

    async def _delayed_analysis(self):
        """Wait briefly then run analysis for better context accumulation."""
        logger.debug(f"[AgendaTracker] Waiting {ANALYSIS_DELAY_SECONDS}s before analysis...")
        await asyncio.sleep(ANALYSIS_DELAY_SECONDS)
        await self._run_analysis()

    async def _run_analysis(self):
        """Run the topic progression analysis."""
        async with self.analysis_lock:
            # Move pending segments to main buffer
            async with self.schedule_lock:
                if not self.pending_segments:
                    logger.debug(f"[AgendaTracker] No pending segments to analyze")
                    return
                segments_to_analyze = self.pending_segments.copy()
                self.pending_segments.clear()

            self.transcript_buffer.extend(segments_to_analyze)

            # Trim buffer to max window
            if len(self.transcript_buffer) > MAX_TRANSCRIPT_WINDOW:
                self.transcript_buffer = self.transcript_buffer[-MAX_TRANSCRIPT_WINDOW:]

            self.last_analysis_time = time.time()
            self.total_analyses += 1

            logger.info(
                f"[AgendaTracker] Running analysis #{self.total_analyses} - "
                f"buffer_size={len(self.transcript_buffer)}, "
                f"current_item={self.current_item_index}"
            )

            await self._analyze_topic_progression()

    async def _analyze_topic_progression(self):
        """
        Use LLM to analyze if the current topic is complete.

        This is the core analysis function that:
        1. Builds context from transcript buffer
        2. Calls LLM with tracking prompt
        3. Parses result and publishes progress if needed
        """
        if not self.agenda or not self.transcript_buffer:
            return

        try:
            # Handle case where meeting hasn't started on any topic yet
            if self.current_item_index == -1:
                await self._detect_first_topic_start()
                return

            # Check if we've already completed all items
            if self.current_item_index >= len(self.agenda.items):
                return

            # Build prompt context
            agenda_items = [
                {
                    "title": item.title,
                    "description": item.description,
                    "estimated_minutes": item.estimated_minutes,
                    "lead_by": item.lead_by,
                }
                for item in self.agenda.items
            ]

            transcript_entries = [
                {
                    "speaker_identity": e.speaker_identity,
                    "speaker": e.speaker_name,
                    "text": e.text,
                }
                for e in self.transcript_buffer[-15:]  # Last 15 segments
            ]

            # Format prompt
            system_prompt, user_prompt = format_tracking_prompt(
                agenda_items=agenda_items,
                current_index=self.current_item_index,
                transcript_entries=transcript_entries,
            )

            # Call LLM
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=system_prompt)
            chat_ctx.add_message(role="user", content=user_prompt)

            response_text = ""
            stream = self.llm.chat(chat_ctx=chat_ctx)
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    response_text += chunk.delta.content

            # Parse response
            result = self._parse_analysis_result(response_text)
            if result:
                logger.info(
                    f"[AgendaTracker] LLM analysis result - "
                    f"complete={result.current_topic_complete}, "
                    f"confidence={result.confidence:.2f}, "
                    f"evidence='{result.evidence[:50]}...'"
                )
                await self._handle_analysis_result(result)
            else:
                logger.warning(f"[AgendaTracker] Failed to parse LLM response: {response_text[:100]}...")

        except Exception as e:
            logger.error(f"Topic progression analysis failed: {e}")

    async def _detect_first_topic_start(self):
        """Detect if the meeting has started discussing the first agenda topic."""
        if not self.agenda or not self.agenda.items:
            return

        # Use simpler heuristic: if we have enough transcript, assume first topic started
        if len(self.transcript_buffer) >= MIN_SEGMENTS_FOR_ANALYSIS:
            self.current_item_index = 0
            self.item_statuses[0] = AgendaItemStatus.IN_PROGRESS
            self.item_start_times[0] = time.time()
            self.segments_since_topic_start = 0

            await self._publish_progress(
                AgendaProgressUpdate(
                    type=AgendaProgressType.TOPIC_STARTED,
                    agenda_id=self.agenda.id,
                    item_index=0,
                    status=AgendaItemStatus.IN_PROGRESS,
                    confidence=0.85,
                    reason="Meeting transcript indicates discussion has begun",
                    transcript_ref=self.transcript_buffer[-1].segment_id if self.transcript_buffer else None,
                )
            )

            logger.info(f"Detected start of first topic: {self.agenda.items[0].title}")

    def _parse_analysis_result(self, response: str) -> Optional[TopicAnalysisResult]:
        """Parse LLM response into TopicAnalysisResult."""
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            if not cleaned:
                return None

            data = json.loads(cleaned)

            return TopicAnalysisResult(
                current_topic_complete=data.get("current_topic_complete", False),
                confidence=float(data.get("confidence", 0.0)),
                evidence=data.get("evidence", "")[:200],
                next_topic_started=data.get("next_topic_started", False),
                next_topic_index=data.get("next_topic_index"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse analysis result: {e}")
            logger.debug(f"Raw response: {response}")
            return None

    async def _handle_analysis_result(self, result: TopicAnalysisResult):
        """Handle the analysis result and publish progress updates if needed."""
        if not self.agenda:
            logger.debug(f"[AgendaTracker] No agenda, skipping result handling")
            return

        # Check confidence threshold
        if result.confidence < MIN_CONFIDENCE_FOR_COMPLETION:
            logger.info(
                f"[AgendaTracker] Confidence {result.confidence:.2f} below threshold "
                f"{MIN_CONFIDENCE_FOR_COMPLETION} - NOT completing topic"
            )
            return

        # Check cooldown to prevent rapid transitions
        now = time.time()
        time_since_transition = now - self.last_transition_time
        if time_since_transition < TRANSITION_COOLDOWN_SECONDS:
            logger.info(
                f"[AgendaTracker] Cooldown active - {time_since_transition:.1f}s since last transition, "
                f"need {TRANSITION_COOLDOWN_SECONDS}s"
            )
            return

        # Handle topic completion
        if result.current_topic_complete:
            logger.info(f"[AgendaTracker] Topic {self.current_item_index} marked complete by LLM, transitioning...")
            await self._complete_current_topic(result)
        else:
            logger.debug(f"[AgendaTracker] LLM says topic not complete yet")

    async def _complete_current_topic(self, result: TopicAnalysisResult):
        """Mark current topic as completed and start next topic."""
        if not self.agenda or self.current_item_index < 0:
            return

        current_idx = self.current_item_index
        current_item = self.agenda.items[current_idx]

        # Create deduplication key
        transition_key = f"{current_idx}-complete"
        if transition_key in self.published_transitions:
            return

        # Mark current topic as completed
        self.item_statuses[current_idx] = AgendaItemStatus.COMPLETED

        # Get transcript ref
        transcript_ref = self.transcript_buffer[-1].segment_id if self.transcript_buffer else None

        # Publish completion
        await self._publish_progress(
            AgendaProgressUpdate(
                type=AgendaProgressType.TOPIC_COMPLETED,
                agenda_id=self.agenda.id,
                item_index=current_idx,
                status=AgendaItemStatus.COMPLETED,
                confidence=result.confidence,
                reason=result.evidence,
                transcript_ref=transcript_ref,
            )
        )

        self.published_transitions.add(transition_key)
        self.last_transition_time = time.time()
        self.successful_transitions += 1

        logger.info(
            f"Topic completed: [{current_idx}] {current_item.title} "
            f"(confidence: {result.confidence:.2f})"
        )

        # Determine next topic
        next_idx = current_idx + 1
        if result.next_topic_started and result.next_topic_index is not None:
            next_idx = result.next_topic_index

        # Start next topic if available
        if next_idx < len(self.agenda.items):
            await self._start_topic(next_idx, result)
        else:
            # All topics completed
            await self._complete_agenda()

    async def _start_topic(self, index: int, trigger_result: Optional[TopicAnalysisResult] = None):
        """Start a new topic."""
        if not self.agenda or index >= len(self.agenda.items):
            return

        old_index = self.current_item_index
        self.current_item_index = index
        self.item_statuses[index] = AgendaItemStatus.IN_PROGRESS
        self.item_start_times[index] = time.time()
        self.segments_since_topic_start = 0

        new_item = self.agenda.items[index]
        transcript_ref = self.transcript_buffer[-1].segment_id if self.transcript_buffer else None

        # Publish topic change if transitioning between topics
        if old_index >= 0 and old_index != index:
            await self._publish_progress(
                AgendaProgressUpdate(
                    type=AgendaProgressType.TOPIC_CHANGE,
                    agenda_id=self.agenda.id,
                    item_index=index,
                    status=AgendaItemStatus.IN_PROGRESS,
                    confidence=trigger_result.confidence if trigger_result else 0.85,
                    reason=f"Transitioned from topic {old_index} to {index}",
                    transcript_ref=transcript_ref,
                )
            )

        # Publish topic started
        await self._publish_progress(
            AgendaProgressUpdate(
                type=AgendaProgressType.TOPIC_STARTED,
                agenda_id=self.agenda.id,
                item_index=index,
                status=AgendaItemStatus.IN_PROGRESS,
                confidence=trigger_result.confidence if trigger_result else 0.85,
                reason=trigger_result.evidence if trigger_result else "Topic started",
                transcript_ref=transcript_ref,
            )
        )

        logger.info(f"Started topic [{index}]: {new_item.title}")

    async def _complete_agenda(self):
        """Mark entire agenda as complete."""
        if not self.agenda:
            return

        await self._publish_progress(
            AgendaProgressUpdate(
                type=AgendaProgressType.AGENDA_COMPLETE,
                agenda_id=self.agenda.id,
                item_index=len(self.agenda.items) - 1,
                status=AgendaItemStatus.COMPLETED,
                confidence=0.95,
                reason="All agenda items have been discussed",
                transcript_ref=self.transcript_buffer[-1].segment_id if self.transcript_buffer else None,
            )
        )

        logger.info(f"Agenda '{self.agenda.id}' completed")

    async def _publish_progress(self, update: AgendaProgressUpdate):
        """Publish progress update to frontend via LiveKit text stream."""
        try:
            # Build payload with camelCase keys for frontend compatibility
            payload = {
                "type": update.type,
                "agendaId": update.agenda_id,
                "itemIndex": update.item_index,
                "status": update.status,
                "confidence": update.confidence,
                "reason": update.reason,
                "transcriptRef": update.transcript_ref,
                "timestamp": update.timestamp,
            }

            await self.room.local_participant.send_text(
                json.dumps(payload),
                topic=AGENDA_PROGRESS_TOPIC,
                attributes={
                    "progress_type": update.type,
                    "item_index": str(update.item_index),
                    "confidence": str(update.confidence),
                },
            )

            logger.info(
                f"Published agenda progress: [{update.type}] "
                f"item {update.item_index}, status {update.status}"
            )

        except Exception as e:
            logger.error(f"Failed to publish agenda progress: {e}")

    # Manual override methods (can be called by agent if needed)

    async def manual_complete_item(self, index: int):
        """Manually mark an item as completed (for manual override support)."""
        if not self.agenda or index >= len(self.agenda.items):
            return

        self.item_statuses[index] = AgendaItemStatus.COMPLETED

        await self._publish_progress(
            AgendaProgressUpdate(
                type=AgendaProgressType.TOPIC_COMPLETED,
                agenda_id=self.agenda.id,
                item_index=index,
                status=AgendaItemStatus.COMPLETED,
                confidence=1.0,
                reason="Manually marked as complete",
            )
        )

        # Start next topic if this was the current one
        if index == self.current_item_index and index + 1 < len(self.agenda.items):
            await self._start_topic(index + 1)

    async def manual_start_item(self, index: int):
        """Manually start an item (for manual override support)."""
        if not self.agenda or index >= len(self.agenda.items):
            return

        # Complete current item if in progress
        if self.current_item_index >= 0 and self.current_item_index != index:
            self.item_statuses[self.current_item_index] = AgendaItemStatus.COMPLETED

        await self._start_topic(index)

    def get_current_item(self) -> Optional[AgendaItem]:
        """Get the currently active agenda item."""
        if not self.agenda or self.current_item_index < 0:
            return None
        if self.current_item_index >= len(self.agenda.items):
            return None
        return self.agenda.items[self.current_item_index]

    def get_progress_percentage(self) -> float:
        """Get overall agenda progress as percentage (0-100)."""
        if not self.agenda or not self.agenda.items:
            return 0.0

        completed = sum(
            1 for idx in range(len(self.agenda.items))
            if self.item_statuses.get(idx) == AgendaItemStatus.COMPLETED
        )
        return (completed / len(self.agenda.items)) * 100
