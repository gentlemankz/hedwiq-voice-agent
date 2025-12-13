"""
Agenda Tracker for Hedwiq Agent - Phase 4 Implementation (Trust-Based LLM)

Provides real-time agenda topic detection and progress tracking.
Analyzes transcripts using LLM to understand conversation context and
detect when discussion moves between agenda topics.

NEW ARCHITECTURE (Trust the LLM):
- Full conversation context given to LLM (not just recent segments)
- No magic constants, confidence thresholds, or stability checks
- LLM decides if speaker has "moved on" to new topic (not keyword matching)
- Trust the LLM's intelligence to understand conversation flow

Philosophy:
    Modern LLMs (GPT-4, Claude, etc.) are intelligent enough to understand
    conversation context. Instead of adding "magic constants" that second-guess
    the LLM, we give it full conversation history and trust its judgment.

Pipeline:
    [Transcript] -> [Buffer] -> [LLM Full Context Analysis] -> [Execute Transition]
                    (all entries)        (~500ms)                 (trust LLM)

Key Features:
- Pure LLM topic detection with FULL conversation context
- No stability checks, no hysteresis, no confidence thresholds
- Participant attributes for late joiner sync
- Graceful degradation on LLM failures
- Automatic first topic start for active agendas

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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from livekit import rtc

from db.agenda import AgendaDB
from schemas.agenda import (
    TopicStartedEvent,
    TopicCompletedEvent,
    TopicSkippedEvent,
    MeetingStartedEvent,
    MeetingEndedEvent,
    AgendaSyncEvent,
    AgendaStateAttribute,
    TopicDetectionResult,
    AGENDA_TOPIC,
    MIN_ANALYSIS_INTERVAL,
    ANALYSIS_DEBOUNCE_SECONDS,
    MAX_TRANSCRIPT_BUFFER,
    MIN_SEGMENT_WORDS_FOR_DETECTION,
)
from prompts.agenda_detection import (
    format_smart_topic_detection_prompt,
    validate_llm_response,
    DETECTION_TIMEOUT_SECONDS,
    DETECTION_MAX_RETRIES,
    DETECTION_TEMPERATURE,
    DETECTION_MAX_TOKENS,
    SMART_DETECTION_REQUIRED_FIELDS,
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
    3. Detects topics using pure LLM context analysis (no word patterns)
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

    # Full transcript buffer - stores COMPLETE conversation for context
    # This is the key change: we give LLM full history, not just recent segments
    transcript_buffer: List[TranscriptEntry] = field(default_factory=list)

    # Analysis control (minimal - just for rate limiting)
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_analysis_time: float = 0
    scheduled_task: Optional[asyncio.Task] = None
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Database client
    db: Optional[AgendaDB] = None

    # Running state
    _running: bool = False

    # Track repeated backward recommendations (sign of LLM confusion)
    _backward_recommendation_count: int = 0
    _last_backward_topic_id: Optional[str] = None

    def __post_init__(self):
        """Initialize locks after dataclass creation."""
        self.analysis_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()
        self.transcript_buffer = []

    async def start(self):
        """
        Start the agenda tracker.

        - Connects to database
        - Loads agenda for the room
        - Auto-starts meeting if agenda is active

        NOTE: This is designed to fail gracefully - if database is unavailable,
        the rest of the agent will still work (transcription, insights, etc.)
        """
        if self._running:
            return

        self._running = True

        # Initialize database connection with graceful failure
        try:
            self.db = AgendaDB()
            await self.db.connect()
        except Exception as e:
            logger.warning(
                f"Failed to connect to database for agenda tracking: {e}. "
                f"Agenda tracking will be disabled for this session."
            )
            self.db = None
            return

        # Load agenda from database
        try:
            await self._load_agenda()
        except Exception as e:
            logger.warning(f"Failed to load agenda: {e}. Agenda tracking disabled.")
            return

        if self.agenda:
            status = self.agenda.get("status", "unknown")
            item_count = self.agenda.get("itemCount", 0)
            logger.info(
                f"AgendaTracker started for room {self.room_id}: "
                f"{item_count} items, status={status}"
            )

            # Check if meeting already started (from a previous session)
            if self.agenda.get("meetingStartedAt"):
                self.is_meeting_started = True
                logger.info(f"Meeting already started (resuming session)")

            # Check if meeting already ended (prevent restarting a completed meeting)
            if self.agenda.get("meetingEndedAt") or status == "completed":
                self.is_meeting_ended = True
                logger.info(f"Meeting already ended, not restarting agenda tracking")
                return

            # AUTO-START: If agenda is active but meeting not started, start it now
            # This removes the "meeting start phrase" gate that was blocking detection
            if status == "active" and not self.is_meeting_started:
                logger.info(f"Active agenda detected, auto-starting meeting and first topic")
                try:
                    await self._handle_meeting_start()
                except Exception as e:
                    logger.error(f"Failed to auto-start meeting: {e}")

            # Sync initial state to participant attributes
            try:
                await self._update_participant_attributes()
            except Exception as e:
                logger.warning(f"Failed to sync participant attributes: {e}")
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
                # Find current item index from existing state
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

        # Skip if no agenda
        if not self.agenda:
            return

        # Skip if meeting already ended
        if self.is_meeting_ended:
            return

        # Log warning but continue if agenda not active (don't silently drop)
        if self.agenda.get("status") != "active":
            logger.debug(f"Agenda status is '{self.agenda.get('status')}', skipping detection")
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
        """Run analysis after debounce delay with exception handling."""
        try:
            await asyncio.sleep(ANALYSIS_DEBOUNCE_SECONDS)
            await self._run_analysis()
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            logger.error(f"Delayed analysis failed: {e}")

    async def _run_analysis(self):
        """Run topic detection analysis on buffered transcripts using LLM."""
        # Check if meeting already ended (prevents race condition with delayed analysis)
        if self.is_meeting_ended:
            logger.debug("Skipping analysis: meeting already ended")
            return

        async with self.analysis_lock:
            # Double-check after acquiring lock
            if self.is_meeting_ended:
                logger.debug("Skipping analysis: meeting ended while waiting for lock")
                return

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
                logger.debug(f"Skipping analysis: only {word_count} words (min: {MIN_SEGMENT_WORDS_FOR_DETECTION})")
                return

            logger.info(f"Running topic analysis on {word_count} words from {len(self.transcript_buffer)} segments")

            try:
                # Use unified LLM detection (no separate meeting start check)
                await self._analyze_with_llm(transcript_text)
            except Exception as e:
                logger.error(f"Topic detection error: {e}", exc_info=True)

    def _build_transcript_text(self) -> str:
        """
        Build FULL conversation transcript from buffer with [RECENT] markers.

        NEW ARCHITECTURE: We give the LLM the complete conversation history,
        not just recent segments. This allows the LLM to understand:
        - The overall flow of the meeting
        - What topics have been discussed vs mentioned
        - The natural progression of the conversation

        FIX: We now mark the last N entries as [RECENT] so the LLM knows
        what's happening NOW vs what happened earlier. This prevents the LLM
        from getting confused and recommending already-completed topics.
        """
        # Use ALL entries - LLM needs full context
        entries = self.transcript_buffer

        if not entries:
            return ""

        # Determine which entries are "recent" (last 10 or 20% of entries, whichever is larger)
        recent_count = max(10, len(entries) // 5)
        recent_start_idx = len(entries) - recent_count

        # Merge adjacent same-speaker entries for cleaner transcript
        # Track whether each merged entry contains any recent segments
        merged = []
        for i, entry in enumerate(entries):
            is_recent = i >= recent_start_idx
            if merged and merged[-1]["identity"] == entry.speaker_identity:
                merged[-1]["text"] += " " + entry.text
                # Mark as recent if ANY segment in this merged entry is recent
                merged[-1]["is_recent"] = merged[-1]["is_recent"] or is_recent
            else:
                merged.append({
                    "identity": entry.speaker_identity,
                    "name": entry.speaker_name,
                    "text": entry.text,
                    "segment_id": entry.segment_id,
                    "is_recent": is_recent,
                })

        # Build transcript with [RECENT] markers for recent entries
        # This helps the LLM focus on what's happening NOW
        lines = []
        for turn in merged:
            prefix = "[RECENT] " if turn.get("is_recent") else ""
            lines.append(f"{prefix}[{turn['name']}]: {turn['text']}")

        return "\n".join(lines)

    # =========================================================================
    # LLM-Based Topic Detection (Pure Context Analysis)
    # =========================================================================

    async def _analyze_with_llm(self, transcript_text: str):
        """
        Use LLM to analyze transcript and detect topic transitions.

        NEW ARCHITECTURE: Trust the LLM completely.
        - Give it full conversation context
        - Ask if speaker has moved to a new topic
        - If LLM says yes, execute the transition
        - No confidence thresholds, no stability checks
        """
        if not self.agenda or self.is_meeting_ended:
            return

        items = self.agenda.get("items", [])
        if not items:
            return

        # Get current item (or None if not started)
        current_item = None
        if 0 <= self.current_item_index < len(items):
            current_item = items[self.current_item_index]

        # Call LLM with full conversation context
        result = await self._llm_smart_detection(
            all_items=items,
            current_item=current_item,
            current_index=self.current_item_index,
            full_transcript=transcript_text
        )

        if not result:
            logger.debug("LLM returned no detection result")
            return

        # Log the result
        logger.info(
            f"LLM analysis: should_transition={result.has_transitioned}, "
            f"new_topic_id={result.next_topic_id}, "
            f"is_meeting_ending={result.is_meeting_ending}, "
            f"reasoning={result.reason}"
        )

        # If LLM detected meeting is ending and we're on the last topic, end the meeting
        if result.is_meeting_ending and not result.has_transitioned:
            next_pending = self._find_next_pending_topic(items)
            if next_pending is None:
                # No more pending topics - this is the last one, end the meeting
                logger.info(f"LLM detected meeting ending on last topic, ending meeting")
                await self._handle_meeting_end()
                return

        # Trust the LLM's decision - if it says transition, we transition
        if result.has_transitioned:
            if not result.next_topic_id:
                # No new topic but transition requested - might be meeting end
                if result.is_meeting_ending:
                    logger.info("LLM recommended transition with meeting ending signal, ending meeting")
                    await self._handle_meeting_end()
                    return
                logger.warning("LLM recommended transition but no new_topic_id provided")
                return

            detected_idx = self._find_item_index(result.next_topic_id)

            if detected_idx < 0:
                logger.warning(f"LLM returned unknown topic_id: {result.next_topic_id}")
                return

            # Basic validation only
            if detected_idx == self.current_item_index:
                logger.debug(f"LLM recommended current topic (idx={detected_idx}), staying")
                return

            # Check if target topic is already completed/skipped
            target_item = items[detected_idx]
            if target_item.get("status") in ("completed", "skipped"):
                logger.info(f"LLM recommended topic {detected_idx} but it's already {target_item.get('status')}")

                # FIX: Track repeated backward recommendations
                # If LLM keeps recommending completed topics, it's confused by the full transcript
                # After enough repeated failures, advance to next pending topic
                if detected_idx <= self.current_item_index:
                    # Track repeated backward recommendations
                    if result.next_topic_id == self._last_backward_topic_id:
                        self._backward_recommendation_count += 1
                    else:
                        self._backward_recommendation_count = 1
                        self._last_backward_topic_id = result.next_topic_id

                    logger.debug(
                        f"Backward recommendation #{self._backward_recommendation_count} "
                        f"for topic {detected_idx}"
                    )

                    # After repeated backward recommendations, trust that meeting has moved on
                    # The LLM is clearly confused - advance to next pending topic
                    if self._backward_recommendation_count >= 3:
                        next_pending = self._find_next_pending_topic(items)
                        if next_pending is not None:
                            next_item = items[next_pending]
                            logger.info(
                                f"LLM repeatedly recommending completed topics "
                                f"(x{self._backward_recommendation_count}), "
                                f"advancing to next pending topic {next_pending}: "
                                f"'{next_item.get('title')}'"
                            )
                            self._backward_recommendation_count = 0
                            self._last_backward_topic_id = None
                            await self._execute_transition(
                                next_pending,
                                reason="repeated_backward_recommendations"
                            )
                            return
                return

            # Reset backward tracking on successful forward transition
            self._backward_recommendation_count = 0
            self._last_backward_topic_id = None

            # Execute the transition - trust the LLM
            logger.info(
                f"TRANSITION: {self.current_item_index} -> {detected_idx} "
                f"(reason: {result.reason})"
            )
            await self._execute_transition(
                detected_idx,
                reason=result.reason or "llm_detected_transition"
            )
        else:
            logger.debug(f"LLM: staying on topic {self.current_item_index} - {result.reason}")

    async def _llm_smart_detection(
        self,
        all_items: List[Dict[str, Any]],
        current_item: Optional[Dict[str, Any]],
        current_index: int,
        full_transcript: str
    ) -> Optional[TopicDetectionResult]:
        """
        Use LLM to detect topic transitions with full conversation context.

        NEW ARCHITECTURE: Simple, trusting approach.
        - Give LLM full conversation
        - Ask if speaker has moved to new topic
        - Trust the response without artificial constraints
        """
        try:
            system_prompt, user_prompt = format_smart_topic_detection_prompt(
                all_items=all_items,
                current_item=current_item,
                current_index=current_index,
                full_transcript=full_transcript
            )

            result = await self._call_llm(
                system_prompt, user_prompt,
                required_fields=SMART_DETECTION_REQUIRED_FIELDS
            )

            if result:
                # Extract simple response
                should_transition = result.get("should_transition", False)
                new_topic_id = result.get("new_topic_id")
                new_topic_index = result.get("new_topic_index")
                reasoning = result.get("reasoning", "")
                current_focus = result.get("current_focus", "")
                is_meeting_ending = result.get("is_meeting_ending", False)

                # If LLM gave index but not ID, try to find ID
                if should_transition and not new_topic_id and new_topic_index is not None:
                    if 0 <= new_topic_index < len(all_items):
                        new_topic_id = all_items[new_topic_index].get("id")

                return TopicDetectionResult(
                    has_transitioned=should_transition,
                    next_topic_id=new_topic_id,
                    confidence=1.0,  # Trust the LLM - no confidence game
                    reason=reasoning,
                    evidence=current_focus,
                    is_meeting_ending=is_meeting_ending,
                )

            return None

        except Exception as e:
            logger.warning(f"LLM smart detection failed: {e}")
            return None

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        required_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make LLM call with timeout, retry, and validation.
        """
        from livekit.agents import llm as lk_llm

        for attempt in range(DETECTION_MAX_RETRIES + 1):
            try:
                chat_ctx = lk_llm.ChatContext()
                chat_ctx.add_message(role="system", content=system_prompt)
                chat_ctx.add_message(role="user", content=user_prompt)

                response_text = ""
                was_truncated = False

                async def get_response():
                    nonlocal response_text, was_truncated
                    # Note: temperature and max_tokens are set via extra_kwargs
                    # as they are not direct parameters of chat()
                    stream = self.llm.chat(
                        chat_ctx=chat_ctx,
                        extra_kwargs={
                            "temperature": DETECTION_TEMPERATURE,
                            "max_tokens": DETECTION_MAX_TOKENS,
                        },
                    )
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            response_text += chunk.delta.content
                            if len(response_text) > DETECTION_MAX_TOKENS * 6 and not was_truncated:
                                logger.warning("LLM response exceeding expected length")
                                was_truncated = True

                await asyncio.wait_for(
                    get_response(),
                    timeout=DETECTION_TIMEOUT_SECONDS
                )

                if was_truncated:
                    logger.warning(f"LLM response was {len(response_text)} chars, attempting parse anyway")

                # Parse JSON response
                cleaned = response_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                try:
                    result = json.loads(cleaned)
                except json.JSONDecodeError:
                    result = self._extract_json_object(cleaned)
                    if result is None:
                        raise json.JSONDecodeError("No valid JSON found", cleaned, 0)

                if required_fields and not validate_llm_response(result, required_fields):
                    logger.warning(f"LLM response missing required fields: {required_fields}")
                    return None

                return result

            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{DETECTION_MAX_RETRIES + 1})")
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

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from text using balanced brace matching."""
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = text.find('{', i + 1)
                        if start == -1:
                            return None
                        depth = 0

        return None

    def _find_item_index(self, item_id: Optional[str]) -> int:
        """Find the index of an item by ID."""
        if not item_id or not self.agenda:
            return -1

        for i, item in enumerate(self.agenda.get("items", [])):
            if item.get("id") == item_id:
                return i

        return -1

    def _find_next_pending_topic(self, items: List[Dict[str, Any]]) -> Optional[int]:
        """Find the next pending topic after the current one."""
        for i in range(self.current_item_index + 1, len(items)):
            if items[i].get("status") == "pending":
                return i
        return None

    # =========================================================================
    # Topic Transitions - Simple, Trust-Based
    # =========================================================================

    async def _execute_transition(
        self,
        next_item_idx: int,
        reason: str
    ):
        """
        Execute a topic transition.

        NEW ARCHITECTURE: No stability checks, no hysteresis, no confidence games.
        If the LLM says transition, we transition.
        """
        if not self.agenda:
            return

        items = self.agenda.get("items", [])
        if next_item_idx >= len(items):
            return

        logger.info(
            f"EXECUTING TRANSITION: {self.current_item_index} -> {next_item_idx} "
            f"(reason: {reason})"
        )

        # Complete current topic if any
        if self.current_item_index >= 0:
            await self._complete_topic(
                self.current_item_index,
                reason=reason
            )

        # Skip any items between current and next (for non-sequential transitions)
        for i in range(self.current_item_index + 1, next_item_idx):
            if items[i].get("status") == "pending":
                await self._skip_topic(i, reason="skipped_in_transition")

        # Start new topic
        await self._start_topic(next_item_idx, reason=reason)

    # =========================================================================
    # Meeting Lifecycle
    # =========================================================================

    async def _handle_meeting_start(self, first_topic_id: Optional[str] = None):
        """Handle meeting start - publish event and start first topic."""
        if self.is_meeting_started:
            return

        self.is_meeting_started = True
        logger.info(f"Meeting started for room {self.room_id}")

        # Update database
        if self.agenda:
            await self.db.start_meeting(self.agenda["id"])

        # Publish meeting_started event
        await self._publish_event(MeetingStartedEvent())

        # Start first topic automatically
        if self.agenda and self.agenda.get("items"):
            await self._start_topic(0, reason="meeting_auto_start")

    async def _handle_meeting_end(self):
        """Handle meeting end - publish event and mark remaining items."""
        if self.is_meeting_ended:
            return

        self.is_meeting_ended = True
        logger.info(f"Meeting ended for room {self.room_id}")

        # Cancel any pending analysis to prevent race conditions
        if self.scheduled_task and not self.scheduled_task.done():
            self.scheduled_task.cancel()
            logger.debug("Cancelled pending analysis task")

        # Complete current topic if any
        if self.current_item_index >= 0 and self.agenda:
            items = self.agenda.get("items", [])
            if self.current_item_index < len(items):
                current_item = items[self.current_item_index]
                if current_item.get("status") == "in_progress":
                    await self._complete_topic(
                        self.current_item_index,
                        reason="meeting_end"
                    )

        # Skip remaining pending items
        if self.agenda:
            for i, item in enumerate(self.agenda.get("items", [])):
                if item.get("status") == "pending":
                    await self._skip_topic(i, reason="meeting_ended")

        # Update database and local state
        if self.agenda:
            await self.db.end_meeting(self.agenda["id"])
            # Update local agenda status to prevent any further processing
            self.agenda["status"] = "completed"

        # Publish meeting_ended event
        await self._publish_event(MeetingEndedEvent())

        # Update participant attributes for late joiner sync
        await self._update_participant_attributes()

    # =========================================================================
    # Topic State Changes
    # =========================================================================

    async def _start_topic(
        self,
        item_idx: int,
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

        # Capture start time for duration tracking
        start_time = datetime.now(timezone.utc)
        start_time_iso = start_time.isoformat()

        # Update local state
        self.current_item_index = item_idx
        item["status"] = "in_progress"
        item["startedAt"] = start_time_iso
        item["startTranscriptRef"] = transcript_ref

        # Update database
        await self.db.update_item_status(item_id, "in_progress", transcript_ref)
        await self.db.update_current_item_index(self.agenda["id"], item_idx)

        # Publish event
        event = TopicStartedEvent(
            item_id=item_id,
            item_index=item_idx,
            transcript_ref=transcript_ref,
            confidence=1.0,  # Trust-based - always confident
        )
        await self._publish_event(event)

        # Update participant attributes
        await self._update_participant_attributes()

        logger.info(f"STARTED TOPIC: '{item.get('title')}' (idx={item_idx}, reason={reason})")

    async def _complete_topic(
        self,
        item_idx: int,
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
        now = datetime.now(timezone.utc)
        if item.get("startedAt"):
            try:
                start_str = item["startedAt"]
                if start_str:
                    start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    actual_duration = int((now - start_time).total_seconds())
            except Exception as e:
                logger.warning(f"Failed to calculate duration: {e}")

        # Update local state
        item["status"] = "completed"
        item["actualDuration"] = actual_duration
        item["completedAt"] = now.isoformat()
        item["endTranscriptRef"] = transcript_ref

        # Update database
        await self.db.update_item_status(
            item_id, "completed", transcript_ref,
            started_at=item.get("startedAt")
        )

        # Publish event
        event = TopicCompletedEvent(
            item_id=item_id,
            item_index=item_idx,
            transcript_ref=transcript_ref,
            confidence=1.0,  # Trust-based - always confident
            actual_duration=actual_duration,
        )
        await self._publish_event(event)

        # Update participant attributes
        await self._update_participant_attributes()

        logger.info(f"COMPLETED TOPIC: '{item.get('title')}' (duration={actual_duration}s, reason={reason})")

        # Auto-end meeting when last topic completed
        await self._check_all_topics_completed()

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

        # Update participant attributes
        await self._update_participant_attributes()

        logger.info(f"SKIPPED TOPIC: '{item.get('title')}' (reason={reason})")

        # Check if all topics done
        await self._check_all_topics_completed()

    async def _check_all_topics_completed(self):
        """Check if all topics are completed/skipped and auto-end meeting."""
        if not self.agenda or self.is_meeting_ended:
            return

        items = self.agenda.get("items", [])
        if not items:
            return

        all_done = all(
            item.get("status") in ("completed", "skipped")
            for item in items
        )

        if all_done:
            logger.info("All agenda topics completed, auto-ending meeting")
            await self._handle_meeting_end()

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

            logger.info(f"Published agenda event: {payload.get('type')}")

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
        Publish full agenda sync event (for late joiners).

        Called when a new participant joins and needs full state.
        """
        if not self.agenda:
            return

        try:
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

            logger.info("Published agenda sync event for late joiner")

        except Exception as e:
            logger.error(f"Failed to publish sync event: {e}")

    # =========================================================================
    # Late Joiner Support
    # =========================================================================

    async def on_participant_connected(self, participant_identity: str):
        """
        Handle new participant connection - publish sync event.

        Called from HedwiqAgent when a new participant joins.
        """
        if not self.agenda or not self.is_meeting_started:
            return

        logger.info(f"New participant connected: {participant_identity}, publishing sync event")
        await self.publish_sync_event()

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
