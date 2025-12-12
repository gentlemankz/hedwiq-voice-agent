"""
Agenda Tracker for Hedwiq Agent - Phase 4 Implementation

Provides real-time agenda topic detection and progress tracking.
Analyzes transcripts to detect when discussion moves between agenda topics
and publishes progress events to the frontend via LiveKit.

Pipeline:
    [Transcript] -> [Pre-filter] -> [Signal Detection] -> [LLM Analysis] -> [Stability Check] -> [Publish]
                     (no LLM)        (keywords/phrases)      (~200ms)         (hysteresis)

Key Features:
- Multi-signal topic detection (explicit mentions, keywords, LLM)
- Stability/hysteresis to prevent thrashing
- Participant attributes for late joiner sync
- Graceful degradation on LLM failures

Usage:
    # In hedwiq_agent.py
    tracker = AgendaTracker(room, room_id, llm)
    await tracker.start()

    # Called from ParticipantTranscriber after final transcript
    await tracker.process_transcript(entry)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from livekit import rtc

from db.agenda import AgendaDB
from schemas.agenda import (
    AgendaItem,
    Agenda,
    AgendaItemStatus,
    AgendaStatus,
    TopicStartedEvent,
    TopicCompletedEvent,
    TopicSkippedEvent,
    MeetingStartedEvent,
    MeetingEndedEvent,
    AgendaSyncEvent,
    AgendaStateAttribute,
    StabilityState,
    TopicDetectionResult,
    AGENDA_TOPIC,
    STABILITY_CONSECUTIVE_K,
    STABILITY_TIME_THRESHOLD,
    SWITCH_CONFIDENCE_THRESHOLD,
    HYSTERESIS_COOLDOWN,
    MIN_ANALYSIS_INTERVAL,
    ANALYSIS_DEBOUNCE_SECONDS,
    MAX_TRANSCRIPT_BUFFER,
    MIN_SEGMENT_WORDS_FOR_DETECTION,
)
from prompts.agenda_detection import (
    format_topic_detection_prompt,
    format_meeting_start_prompt,
    format_meeting_end_prompt,
    EXPLICIT_TRANSITION_PATTERNS,
    MEETING_START_PATTERNS,
    MEETING_END_PATTERNS,
    MIN_DETECTION_CONFIDENCE,
    DETECTION_TIMEOUT_SECONDS,
    DETECTION_MAX_RETRIES,
    DETECTION_TEMPERATURE,
    DETECTION_MAX_TOKENS,
)
from insight_analyzer import TranscriptEntry

logger = logging.getLogger("hedwiq-agenda-tracker")


@dataclass
class AgendaTracker:
    """
    Real-time agenda topic detection and progress tracking.

    This class:
    1. Loads agenda from database on room join
    2. Receives transcript segments from ParticipantTranscriber
    3. Detects topic transitions using multi-signal approach
    4. Updates database with status changes
    5. Publishes events to LiveKit for frontend updates
    6. Updates participant attributes for late joiner sync
    """

    room: rtc.Room
    room_id: str
    llm: Any  # OpenAILLM from livekit.plugins.openai

    # State
    agenda: Optional[Dict[str, Any]] = None
    current_item_index: int = -1  # -1 = not started
    is_meeting_started: bool = False
    is_meeting_ended: bool = False

    # Transcript buffer
    transcript_buffer: List[TranscriptEntry] = field(default_factory=list)

    # Stability tracking
    stability: StabilityState = field(default_factory=StabilityState)

    # Analysis control
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_analysis_time: float = 0
    scheduled_task: Optional[asyncio.Task] = None
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Database client
    db: Optional[AgendaDB] = None

    # Running state
    _running: bool = False

    def __post_init__(self):
        """Initialize locks after dataclass creation."""
        self.analysis_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()
        self.transcript_buffer = []
        self.stability = StabilityState()

    async def start(self):
        """
        Start the agenda tracker.

        - Connects to database
        - Loads agenda for the room
        - Initializes state
        """
        if self._running:
            return

        self._running = True

        # Initialize database connection
        self.db = AgendaDB()
        await self.db.connect()

        # Load agenda from database
        await self._load_agenda()

        if self.agenda:
            logger.info(
                f"AgendaTracker started for room {self.room_id}: "
                f"{self.agenda['itemCount']} items, status={self.agenda['status']}"
            )
            # Sync initial state to participant attributes
            await self._update_participant_attributes()
        else:
            logger.info(f"AgendaTracker started for room {self.room_id}: No agenda found")

    async def stop(self):
        """Stop the agenda tracker and clean up."""
        if not self._running:
            return

        self._running = False

        # Cancel any pending analysis
        if self.scheduled_task and not self.scheduled_task.done():
            self.scheduled_task.cancel()
            try:
                await self.scheduled_task
            except asyncio.CancelledError:
                pass

        # Close database connection
        if self.db:
            await self.db.close()

        logger.info(f"AgendaTracker stopped for room {self.room_id}")

    async def _load_agenda(self):
        """Load agenda from database."""
        try:
            self.agenda = await self.db.get_agenda_for_room(self.room_id)

            if self.agenda:
                # Extract keywords for each item
                for item in self.agenda.get("items", []):
                    item["keywords"] = self._extract_keywords(
                        item.get("title", ""),
                        item.get("description", "")
                    )

                # Check if meeting already started (from a previous session)
                if self.agenda.get("meetingStartedAt"):
                    self.is_meeting_started = True

                # Find current item index
                current_idx = self.agenda.get("currentItemIndex")
                if current_idx is not None:
                    self.current_item_index = current_idx
                else:
                    # Find first non-completed item
                    for i, item in enumerate(self.agenda.get("items", [])):
                        if item.get("status") == "in_progress":
                            self.current_item_index = i
                            break

        except Exception as e:
            logger.error(f"Failed to load agenda: {e}")
            self.agenda = None

    def _extract_keywords(self, title: str, description: Optional[str]) -> List[str]:
        """Extract keywords from title and description for matching."""
        text = title.lower()
        if description:
            text += " " + description.lower()

        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "it", "its", "we", "our", "us", "about"
        }

        words = text.split()
        keywords = [
            w.strip(".,;:!?()[]{}\"'")
            for w in words
            if len(w) > 2 and w.lower() not in stop_words
        ]
        return list(set(keywords))

    # =========================================================================
    # Transcript Processing
    # =========================================================================

    async def process_transcript(self, entry: TranscriptEntry):
        """
        Process a transcript entry for topic detection.

        Called from ParticipantTranscriber after each final transcript.

        Args:
            entry: TranscriptEntry with speaker info, text, and segment ID
        """
        if not self._running or not entry.is_final:
            return

        # Skip if no agenda or agenda not active
        if not self.agenda or self.agenda.get("status") != "active":
            return

        # Add to buffer
        async with self.schedule_lock:
            self.transcript_buffer.append(entry)

            # Trim buffer if too long
            if len(self.transcript_buffer) > MAX_TRANSCRIPT_BUFFER:
                self.transcript_buffer = self.transcript_buffer[-MAX_TRANSCRIPT_BUFFER:]

            # Check if we should schedule analysis
            now = time.time()
            time_since_last = now - self.last_analysis_time
            should_schedule = (
                len(self.transcript_buffer) >= 3 or
                time_since_last >= MIN_ANALYSIS_INTERVAL
            )

            if should_schedule and (
                self.scheduled_task is None or self.scheduled_task.done()
            ):
                self.scheduled_task = asyncio.create_task(self._delayed_analysis())

    async def _delayed_analysis(self):
        """Run analysis after debounce delay."""
        await asyncio.sleep(ANALYSIS_DEBOUNCE_SECONDS)
        await self._run_analysis()

    async def _run_analysis(self):
        """Run topic detection analysis on buffered transcripts."""
        async with self.analysis_lock:
            if not self.transcript_buffer:
                return

            self.last_analysis_time = time.time()

            # Build combined transcript text
            transcript_text = self._build_transcript_text()

            if not transcript_text.strip():
                return

            # Pre-filter: skip very short segments
            word_count = len(transcript_text.split())
            if word_count < MIN_SEGMENT_WORDS_FOR_DETECTION:
                return

            try:
                if not self.is_meeting_started:
                    await self._check_meeting_start(transcript_text)
                elif not self.is_meeting_ended:
                    await self._check_topic_transition(transcript_text)
            except Exception as e:
                logger.error(f"Topic detection error: {e}")

    def _build_transcript_text(self) -> str:
        """Build combined transcript text from buffer."""
        # Merge adjacent same-speaker entries
        merged = []
        for entry in self.transcript_buffer[-15:]:  # Last 15 entries
            if merged and merged[-1]["identity"] == entry.speaker_identity:
                merged[-1]["text"] += " " + entry.text
            else:
                merged.append({
                    "identity": entry.speaker_identity,
                    "name": entry.speaker_name,
                    "text": entry.text,
                    "segment_id": entry.segment_id,
                })

        lines = []
        for turn in merged:
            lines.append(f"[{turn['name']}]: {turn['text']}")

        return "\n".join(lines)

    # =========================================================================
    # Meeting Start/End Detection
    # =========================================================================

    async def _check_meeting_start(self, transcript_text: str):
        """Check if meeting has started discussing agenda."""
        # Signal 1: Explicit start phrases
        text_lower = transcript_text.lower()
        for pattern in MEETING_START_PATTERNS:
            if re.search(pattern, text_lower):
                logger.info("Detected explicit meeting start phrase")
                await self._handle_meeting_start()
                return

        # Signal 2: LLM analysis (if first agenda item keywords present)
        if self.agenda and self.agenda.get("items"):
            first_item = self.agenda["items"][0]
            first_keywords = first_item.get("keywords", [])

            # Check keyword overlap
            text_words = set(text_lower.split())
            keyword_overlap = len(text_words & set(first_keywords))

            if keyword_overlap >= 2:
                # Try LLM confirmation
                result = await self._llm_meeting_start_check(transcript_text)
                if result and result.get("has_started") and result.get("confidence", 0) >= MIN_DETECTION_CONFIDENCE:
                    logger.info(f"LLM detected meeting start: {result.get('evidence', '')}")
                    await self._handle_meeting_start(result.get("first_topic_id"))
                    return

    async def _handle_meeting_start(self, first_topic_id: Optional[str] = None):
        """Handle meeting start - publish event and start first topic."""
        if self.is_meeting_started:
            return

        self.is_meeting_started = True

        # Update database
        if self.agenda:
            await self.db.start_meeting(self.agenda["id"])

        # Publish meeting_started event
        await self._publish_event(MeetingStartedEvent())

        # Start first topic
        if self.agenda and self.agenda.get("items"):
            first_item = self.agenda["items"][0]
            await self._start_topic(0, confidence=0.9, reason="meeting_start")

    async def _check_meeting_end(self, transcript_text: str):
        """Check if meeting is ending."""
        # Only check if all/most items completed
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        completed_count = sum(1 for i in items if i.get("status") in ("completed", "skipped"))

        # Require at least 50% completed to consider end
        if completed_count < len(items) * 0.5:
            return

        # Signal 1: Explicit end phrases
        text_lower = transcript_text.lower()
        for pattern in MEETING_END_PATTERNS:
            if re.search(pattern, text_lower):
                logger.info("Detected explicit meeting end phrase")
                await self._handle_meeting_end()
                return

    async def _handle_meeting_end(self):
        """Handle meeting end - publish event and mark remaining items."""
        if self.is_meeting_ended:
            return

        self.is_meeting_ended = True

        # Complete current topic if any
        if self.current_item_index >= 0 and self.agenda:
            items = self.agenda.get("items", [])
            if self.current_item_index < len(items):
                current_item = items[self.current_item_index]
                if current_item.get("status") == "in_progress":
                    await self._complete_topic(
                        self.current_item_index,
                        confidence=0.8,
                        reason="meeting_end"
                    )

        # Skip remaining pending items
        if self.agenda:
            for i, item in enumerate(self.agenda.get("items", [])):
                if item.get("status") == "pending":
                    await self._skip_topic(i, reason="meeting_ended")

        # Update database
        if self.agenda:
            await self.db.end_meeting(self.agenda["id"])

        # Publish meeting_ended event
        await self._publish_event(MeetingEndedEvent())

    # =========================================================================
    # Topic Transition Detection
    # =========================================================================

    async def _check_topic_transition(self, transcript_text: str):
        """Check for topic transitions using multi-signal approach."""
        if self.current_item_index < 0:
            # No current topic - shouldn't happen after meeting start
            return

        items = self.agenda.get("items", [])
        if self.current_item_index >= len(items):
            # Check for meeting end
            await self._check_meeting_end(transcript_text)
            return

        current_item = items[self.current_item_index]

        # Get upcoming items (pending ones)
        upcoming_items = [
            item for item in items[self.current_item_index + 1:]
            if item.get("status") == "pending"
        ]

        if not upcoming_items:
            # No more items - check for meeting end
            await self._check_meeting_end(transcript_text)
            return

        text_lower = transcript_text.lower()

        # Signal 1: Explicit transition phrases (highest confidence)
        explicit_result = self._check_explicit_transitions(text_lower, upcoming_items)
        if explicit_result:
            next_item_idx = self._find_item_index(explicit_result["item_id"])
            if next_item_idx >= 0:
                await self._try_transition(
                    next_item_idx,
                    confidence=0.95,
                    reason="explicit_mention"
                )
                return

        # Signal 2: Keyword matching (medium confidence)
        keyword_result = self._check_keyword_match(text_lower, upcoming_items)

        # Signal 3: LLM analysis (if signals 1-2 suggest potential transition)
        if keyword_result and keyword_result["score"] > 0.5:
            llm_result = await self._llm_topic_detection(
                current_item,
                upcoming_items[:3],  # Top 3 upcoming
                transcript_text
            )

            if llm_result and llm_result.has_transitioned:
                # Combine signals
                combined_confidence = self._combine_signals(
                    explicit=0.0,  # No explicit match
                    keyword=keyword_result["score"],
                    llm=llm_result.confidence
                )

                if combined_confidence >= SWITCH_CONFIDENCE_THRESHOLD:
                    next_item_idx = self._find_item_index(llm_result.next_topic_id)
                    if next_item_idx >= 0:
                        await self._try_transition(
                            next_item_idx,
                            confidence=combined_confidence,
                            reason=llm_result.reason or "llm_detection"
                        )
                        return

    def _check_explicit_transitions(
        self,
        text: str,
        upcoming_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for explicit transition phrases."""
        for pattern in EXPLICIT_TRANSITION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                mentioned_topic = match.group(match.lastindex) if match.lastindex else ""

                # Find matching upcoming item
                for item in upcoming_items:
                    title = item.get("title", "").lower()
                    similarity = self._fuzzy_match(mentioned_topic.lower(), title)
                    if similarity > 0.5:
                        return {"item_id": item["id"], "similarity": similarity}

        return None

    def _check_keyword_match(
        self,
        text: str,
        upcoming_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check keyword overlap with upcoming items."""
        text_words = set(text.split())
        best_match = None
        best_score = 0.0

        for item in upcoming_items:
            keywords = set(item.get("keywords", []))
            if not keywords:
                continue

            overlap = len(text_words & keywords)
            score = overlap / len(keywords) if keywords else 0

            if score > best_score:
                best_score = score
                best_match = {"item_id": item["id"], "score": min(score, 0.85)}

        return best_match if best_score > 0.3 else None

    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Simple fuzzy matching based on word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _combine_signals(
        self,
        explicit: float,
        keyword: float,
        llm: float
    ) -> float:
        """Combine detection signals with weighted average."""
        # Weights: explicit > LLM > keyword
        weights = {"explicit": 0.4, "llm": 0.35, "keyword": 0.25}

        if explicit > 0:
            return explicit  # Explicit match overrides others

        combined = (
            weights["keyword"] * keyword +
            weights["llm"] * llm
        )

        return min(combined, 1.0)

    def _find_item_index(self, item_id: Optional[str]) -> int:
        """Find the index of an item by ID."""
        if not item_id or not self.agenda:
            return -1

        for i, item in enumerate(self.agenda.get("items", [])):
            if item.get("id") == item_id:
                return i

        return -1

    # =========================================================================
    # Stability / Hysteresis
    # =========================================================================

    async def _try_transition(
        self,
        next_item_idx: int,
        confidence: float,
        reason: str
    ):
        """
        Try to transition to a new topic with stability checks.

        Implements hysteresis to prevent rapid topic switching.
        """
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        if next_item_idx >= len(items):
            return

        next_item = items[next_item_idx]
        next_item_id = next_item.get("id")

        now = time.time()

        # Hysteresis: don't switch too soon after last switch
        if now - self.stability.last_switch_time < HYSTERESIS_COOLDOWN:
            logger.debug(f"Hysteresis cooldown active, skipping transition")
            return

        # Confidence check
        if confidence < SWITCH_CONFIDENCE_THRESHOLD:
            logger.debug(f"Confidence {confidence:.2f} below threshold, skipping")
            return

        # Stability check: same prediction K times in a row OR T seconds
        if next_item_id == self.stability.last_predicted_topic:
            self.stability.consecutive_predictions += 1
        else:
            self.stability.consecutive_predictions = 1
            self.stability.first_prediction_time = now
            self.stability.last_predicted_topic = next_item_id

        time_stable = (now - self.stability.first_prediction_time) >= STABILITY_TIME_THRESHOLD
        count_stable = self.stability.consecutive_predictions >= STABILITY_CONSECUTIVE_K

        if not (time_stable or count_stable):
            logger.debug(
                f"Stability check failed: consecutive={self.stability.consecutive_predictions}, "
                f"time_stable={time_stable}"
            )
            return

        # Commit the transition
        logger.info(
            f"Topic transition: {self.current_item_index} -> {next_item_idx} "
            f"(conf={confidence:.2f}, reason={reason})"
        )

        # Complete current topic
        if self.current_item_index >= 0:
            await self._complete_topic(
                self.current_item_index,
                confidence=confidence,
                reason=reason
            )

        # Skip any items between current and next
        for i in range(self.current_item_index + 1, next_item_idx):
            if items[i].get("status") == "pending":
                await self._skip_topic(i, reason="skipped_in_transition")

        # Start new topic
        await self._start_topic(next_item_idx, confidence=confidence, reason=reason)

        # Update stability tracking
        self.stability.last_switch_time = now
        self.stability.consecutive_predictions = 0
        self.stability.last_predicted_topic = None

    # =========================================================================
    # Topic State Changes
    # =========================================================================

    async def _start_topic(
        self,
        item_idx: int,
        confidence: float,
        reason: str
    ):
        """Start a topic and publish event."""
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        if item_idx >= len(items):
            return

        item = items[item_idx]
        item_id = item.get("id")
        transcript_ref = self.transcript_buffer[-1].segment_id if self.transcript_buffer else None

        # Update local state
        self.current_item_index = item_idx
        item["status"] = "in_progress"

        # Update database
        await self.db.update_item_status(item_id, "in_progress", transcript_ref)
        await self.db.update_current_item_index(self.agenda["id"], item_idx)

        # Publish event
        event = TopicStartedEvent(
            item_id=item_id,
            item_index=item_idx,
            transcript_ref=transcript_ref,
            confidence=confidence,
        )
        await self._publish_event(event)

        # Update participant attributes
        await self._update_participant_attributes()

        logger.info(f"Started topic: {item.get('title')} (idx={item_idx})")

    async def _complete_topic(
        self,
        item_idx: int,
        confidence: float,
        reason: str
    ):
        """Complete a topic and publish event."""
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        if item_idx >= len(items):
            return

        item = items[item_idx]
        item_id = item.get("id")
        transcript_ref = self.transcript_buffer[-1].segment_id if self.transcript_buffer else None

        # Calculate duration
        actual_duration = 0
        if item.get("startedAt"):
            try:
                from datetime import datetime
                start_str = item["startedAt"]
                if start_str:
                    start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    actual_duration = int((datetime.utcnow() - start_time.replace(tzinfo=None)).total_seconds())
            except Exception:
                pass

        # Update local state
        item["status"] = "completed"
        item["actualDuration"] = actual_duration

        # Update database
        await self.db.update_item_status(item_id, "completed", transcript_ref)

        # Publish event
        event = TopicCompletedEvent(
            item_id=item_id,
            item_index=item_idx,
            transcript_ref=transcript_ref,
            confidence=confidence,
            actual_duration=actual_duration,
        )
        await self._publish_event(event)

        # Update participant attributes
        await self._update_participant_attributes()

        logger.info(f"Completed topic: {item.get('title')} (duration={actual_duration}s)")

    async def _skip_topic(self, item_idx: int, reason: str):
        """Skip a topic and publish event."""
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        if item_idx >= len(items):
            return

        item = items[item_idx]
        item_id = item.get("id")

        # Update local state
        item["status"] = "skipped"

        # Update database
        await self.db.update_item_status(item_id, "skipped")

        # Publish event
        event = TopicSkippedEvent(
            item_id=item_id,
            item_index=item_idx,
            reason=reason,
        )
        await self._publish_event(event)

        logger.info(f"Skipped topic: {item.get('title')} (reason={reason})")

    # =========================================================================
    # LLM Detection
    # =========================================================================

    async def _llm_meeting_start_check(self, transcript_text: str) -> Optional[Dict[str, Any]]:
        """Use LLM to check if meeting has started."""
        if not self.agenda:
            return None

        try:
            system_prompt, user_prompt = format_meeting_start_prompt(
                self.agenda.get("items", []),
                transcript_text
            )

            result = await self._call_llm(system_prompt, user_prompt)
            return result

        except Exception as e:
            logger.warning(f"LLM meeting start check failed: {e}")
            return None

    async def _llm_topic_detection(
        self,
        current_item: Dict[str, Any],
        upcoming_items: List[Dict[str, Any]],
        transcript_text: str
    ) -> Optional[TopicDetectionResult]:
        """Use LLM to detect topic transitions."""
        try:
            system_prompt, user_prompt = format_topic_detection_prompt(
                current_item,
                upcoming_items,
                transcript_text
            )

            result = await self._call_llm(system_prompt, user_prompt)

            if result:
                return TopicDetectionResult(
                    has_transitioned=result.get("has_transitioned", False),
                    next_topic_id=result.get("next_topic_id"),
                    confidence=float(result.get("confidence", 0)),
                    reason=result.get("reason"),
                    evidence=result.get("evidence"),
                )

            return None

        except Exception as e:
            logger.warning(f"LLM topic detection failed: {e}")
            return None

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Make LLM call with timeout and retry."""
        from livekit.agents import llm as lk_llm

        for attempt in range(DETECTION_MAX_RETRIES + 1):
            try:
                chat_ctx = lk_llm.ChatContext()
                chat_ctx.add_message(role="system", content=system_prompt)
                chat_ctx.add_message(role="user", content=user_prompt)

                response_text = ""

                async def get_response():
                    nonlocal response_text
                    stream = self.llm.chat(chat_ctx=chat_ctx)
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            response_text += chunk.delta.content

                await asyncio.wait_for(
                    get_response(),
                    timeout=DETECTION_TIMEOUT_SECONDS
                )

                # Parse JSON response
                cleaned = response_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                return json.loads(cleaned)

            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
                if attempt < DETECTION_MAX_RETRIES:
                    continue
                return None

            except json.JSONDecodeError as e:
                logger.warning(f"LLM response parse failed: {e}")
                return None

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                if attempt < DETECTION_MAX_RETRIES:
                    await asyncio.sleep(0.1)
                    continue
                return None

        return None

    # =========================================================================
    # Event Publishing
    # =========================================================================

    async def _publish_event(self, event: Any):
        """Publish agenda event via LiveKit text stream."""
        try:
            payload = event.to_dict()

            await self.room.local_participant.send_text(
                json.dumps(payload),
                topic=AGENDA_TOPIC,
                attributes={
                    "event_type": payload.get("type", "unknown"),
                    "item_id": str(payload.get("itemId", "")),
                },
            )

            logger.debug(f"Published agenda event: {payload.get('type')}")

        except Exception as e:
            logger.error(f"Failed to publish agenda event: {e}")

    async def _update_participant_attributes(self):
        """
        Update agent's participant attributes with current agenda state.

        This enables late joiner sync without relying on text stream replay.
        """
        if not self.agenda:
            return

        try:
            # Build compact state
            completed_ids = [
                item.get("id")
                for item in self.agenda.get("items", [])
                if item.get("status") in ("completed", "skipped")
            ]

            current_id = None
            if 0 <= self.current_item_index < len(self.agenda.get("items", [])):
                current_id = self.agenda["items"][self.current_item_index].get("id")

            state = AgendaStateAttribute(
                v=self.agenda.get("version", 1),
                c=current_id,
                d=completed_ids,
                s=int(time.time()) if self.is_meeting_started else None,
            )

            # Update participant attributes
            await self.room.local_participant.set_attributes({
                "agendaState": json.dumps(state.to_dict())
            })

            logger.debug(f"Updated participant attributes with agenda state")

        except Exception as e:
            logger.warning(f"Failed to update participant attributes: {e}")

    async def publish_sync_event(self):
        """
        Publish full agenda sync event (for late joiners requesting sync).

        Called when a new participant joins and needs full state.
        """
        if not self.agenda:
            return

        try:
            # Build agenda dict for sync (convert to frontend format)
            agenda_data = {
                "id": self.agenda.get("id"),
                "roomId": self.agenda.get("roomId"),
                "createdBy": self.agenda.get("createdBy"),
                "itemCount": self.agenda.get("itemCount"),
                "status": self.agenda.get("status"),
                "currentItemIndex": self.current_item_index if self.current_item_index >= 0 else None,
                "version": self.agenda.get("version"),
                "meetingStartedAt": self.agenda.get("meetingStartedAt"),
                "meetingEndedAt": self.agenda.get("meetingEndedAt"),
                "items": self.agenda.get("items", []),
            }

            event = AgendaSyncEvent(
                agenda=agenda_data,
                current_item_index=self.current_item_index if self.current_item_index >= 0 else None,
            )

            await self._publish_event(event)

            logger.info("Published agenda sync event")

        except Exception as e:
            logger.error(f"Failed to publish sync event: {e}")

    # =========================================================================
    # Public Methods
    # =========================================================================

    def has_agenda(self) -> bool:
        """Check if this room has an active agenda."""
        return (
            self.agenda is not None and
            self.agenda.get("status") == "active"
        )

    def get_current_topic(self) -> Optional[Dict[str, Any]]:
        """Get the current active topic."""
        if not self.agenda or self.current_item_index < 0:
            return None

        items = self.agenda.get("items", [])
        if self.current_item_index < len(items):
            return items[self.current_item_index]

        return None

    def get_progress_percentage(self) -> float:
        """Get meeting progress as percentage."""
        if not self.agenda:
            return 0.0

        items = self.agenda.get("items", [])
        if not items:
            return 0.0

        completed = sum(1 for i in items if i.get("status") in ("completed", "skipped"))
        return (completed / len(items)) * 100
