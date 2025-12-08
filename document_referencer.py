"""
Document Referencer for Hedwiq Agent - Phase 2 Implementation

Provides real-time document reference detection using hybrid retrieval.
Integrates with hedwiq_agent.py's VAD/transcript flow.

Phase 2 implements:
- Pre-filter (no LLM) to skip irrelevant segments
- Hybrid retrieval (BM25 + embeddings + RRF)
- Candidate publishing for frontend preview
- Deduplication with TTL

Phase 3 will add:
- Single LLM alignment step for validation
- High-confidence reference publishing

Usage:
    # In hedwiq_agent.py
    referencer = DocumentReferencer(room, room_id, document_store)
    await referencer.start()

    # Called from ParticipantTranscriber after final transcript
    await referencer.on_transcript_final(
        segment_id="user1-123",
        transcript="As mentioned in the Q4 report, revenue increased...",
        speaker_identity="user1",
        duration_seconds=3.5
    )
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from livekit import rtc

from hybrid_retriever import (
    HybridRetriever,
    RoomRetrieverManager,
    RetrievalMetrics,
)
from persistent_store import PersistentDocumentStore
from schemas.documents import (
    RetrievalCandidate,
    DEDUPE_TTL_MINUTES,
)

logger = logging.getLogger("hedwiq-document-referencer")

# LiveKit text stream topic for document reference candidates
DOCUMENT_REFERENCE_TOPIC = "hedwiq.document_reference"
DOCUMENT_CANDIDATE_TOPIC = "hedwiq.document_candidate"  # Phase 2: candidates for preview


@dataclass
class ReferenceDedupe:
    """Track recent references for deduplication."""
    fingerprint: str
    timestamp: float


class DocumentReferencer:
    """
    Real-time document reference detection using hybrid retrieval.

    This class integrates with hedwiq_agent.py to:
    1. Receive final transcript segments from VAD flow
    2. Apply lightweight pre-filter (no LLM)
    3. Run hybrid retrieval (BM25 + embeddings)
    4. Publish candidates for frontend preview
    5. (Phase 3) Run LLM alignment for high-confidence references

    Thread-safe for concurrent transcript processing.
    """

    def __init__(
        self,
        room: rtc.Room,
        room_id: str,
        document_store: PersistentDocumentStore,
        retriever_manager: Optional[RoomRetrieverManager] = None,
    ):
        """
        Initialize document referencer.

        Args:
            room: LiveKit room for publishing references
            room_id: Room ID for scoping
            document_store: Document store for accessing documents
            retriever_manager: Optional pre-initialized retriever manager
        """
        self.room = room
        self.room_id = room_id
        self._store = document_store

        # Get or create retriever manager
        if retriever_manager:
            self._retriever_manager = retriever_manager
        else:
            self._retriever_manager = RoomRetrieverManager.get_instance(document_store)

        # Processing queue for async handling
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

        # Deduplication cache: fingerprint -> timestamp
        self._recent_refs: Dict[str, float] = {}
        self._dedupe_lock = asyncio.Lock()

        # Metrics
        self._metrics = RetrievalMetrics(room_id)

        # State
        self._running = False

    async def start(self):
        """
        Start the document referencer.

        Builds initial retrieval index if documents exist.
        """
        if self._running:
            return

        self._running = True

        # Build initial index for this room
        self._retriever_manager.rebuild_room_index(self.room_id)

        # Start processing loop
        self._task = asyncio.create_task(self._process_queue())

        logger.info(f"DocumentReferencer started for room {self.room_id}")

    async def stop(self):
        """Stop the document referencer and log metrics."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Log final metrics
        self._metrics.log_summary()

        logger.info(f"DocumentReferencer stopped for room {self.room_id}")

    async def on_transcript_final(
        self,
        segment_id: str,
        transcript: str,
        speaker_identity: str,
        duration_seconds: float
    ):
        """
        Called when a final transcript segment is received from VAD.

        This is the main entry point from hedwiq_agent.py's ParticipantTranscriber.

        Args:
            segment_id: Unique segment ID (e.g., "user1-123")
            transcript: The transcript text
            speaker_identity: Speaker's LiveKit identity
            duration_seconds: Duration of the speech segment
        """
        if not self._running:
            return

        # Check if we have any documents for this room
        if not self._retriever_manager.has_documents(self.room_id):
            return

        # Queue for async processing
        await self._queue.put({
            "segment_id": segment_id,
            "transcript": transcript,
            "speaker": speaker_identity,
            "duration": duration_seconds,
            "timestamp": time.time(),
        })

    async def on_document_added(self):
        """
        Called when a document is uploaded to this room.

        Triggers index rebuild.
        """
        self._retriever_manager.rebuild_room_index(self.room_id)
        logger.info(f"Rebuilt retrieval index for room {self.room_id} after document upload")

    async def on_document_removed(self):
        """
        Called when a document is removed from this room.

        Triggers index rebuild.
        """
        self._retriever_manager.rebuild_room_index(self.room_id)
        logger.info(f"Rebuilt retrieval index for room {self.room_id} after document removal")

    async def _process_queue(self):
        """Process transcript segments from queue."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_segment(item)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing segment: {e}")

    async def _process_segment(self, item: dict):
        """
        Process a single transcript segment through the retrieval pipeline.

        Pipeline:
        1. Pre-filter (skip short/filler segments)
        2. Hybrid retrieval (BM25 + embeddings + RRF)
        3. Deduplication
        4. Publish candidates
        """
        transcript = item["transcript"]
        segment_id = item["segment_id"]
        duration = item["duration"]
        speaker = item["speaker"]

        # Get retriever for this room
        retriever = self._retriever_manager.get_retriever(self.room_id)
        if not retriever:
            return

        # Step 1: Pre-filter (no LLM)
        prefilter_result = retriever.prefilter_segment(transcript, duration)
        self._metrics.record_prefilter(
            prefilter_result.should_process,
            prefilter_result.reason
        )

        if not prefilter_result.should_process:
            return

        # Step 2: Hybrid retrieval
        start_time = time.time()
        candidates = retriever.retrieve(transcript, top_k=3)
        latency_ms = (time.time() - start_time) * 1000

        self._metrics.record_retrieval(latency_ms, candidates)

        if not candidates:
            return

        # Step 3: Deduplication
        deduplicated = await self._deduplicate_candidates(segment_id, candidates)

        if not deduplicated:
            return

        # Step 4: Publish candidates (Phase 2)
        # In Phase 3, this would go to LLM alignment first
        await self._publish_candidates(segment_id, transcript, speaker, deduplicated)

    async def _deduplicate_candidates(
        self,
        segment_id: str,
        candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        """
        Deduplicate candidates based on segment+section fingerprint.

        Prevents the same document section from being referenced
        multiple times within the TTL window.
        """
        async with self._dedupe_lock:
            current_time = time.time()
            ttl_seconds = DEDUPE_TTL_MINUTES * 60

            # Clean old entries
            self._recent_refs = {
                fp: ts for fp, ts in self._recent_refs.items()
                if current_time - ts < ttl_seconds
            }

            # Filter duplicates
            unique_candidates = []
            for candidate in candidates:
                fingerprint = f"{segment_id}:{candidate.segment_id}"

                if fingerprint not in self._recent_refs:
                    self._recent_refs[fingerprint] = current_time
                    unique_candidates.append(candidate)
                else:
                    logger.debug(f"Skipping duplicate: {fingerprint}")

            return unique_candidates

    async def _publish_candidates(
        self,
        segment_id: str,
        transcript: str,
        speaker: str,
        candidates: List[RetrievalCandidate]
    ):
        """
        Publish retrieval candidates via LiveKit text stream.

        Phase 2: Publishes candidates for frontend preview.
        Phase 3: Would first go through LLM alignment.

        Args:
            segment_id: Transcript segment ID
            transcript: Original transcript text
            speaker: Speaker identity
            candidates: Retrieved candidates
        """
        try:
            # Build payload for frontend
            payload = {
                "type": "retrieval_candidates",
                "segment_id": segment_id,
                "transcript": transcript[:200],  # Truncate for preview
                "speaker": speaker,
                "timestamp": int(time.time() * 1000),
                "candidates": [
                    {
                        "segment_id": c.segment_id,
                        "document_id": c.document_id,
                        "page_number": c.page_number,
                        "section_title": c.section_title,
                        "content": c.content[:300],  # Truncate for preview
                        "score": round(c.score, 4),
                        "bbox": c.bbox.to_dict() if c.bbox else None,
                    }
                    for c in candidates
                ],
            }

            await self.room.local_participant.send_text(
                json.dumps(payload),
                topic=DOCUMENT_CANDIDATE_TOPIC,
                attributes={
                    "segment_id": segment_id,
                    "candidate_count": str(len(candidates)),
                    "top_score": str(round(candidates[0].score, 4)) if candidates else "0",
                },
            )

            logger.info(
                f"Published {len(candidates)} candidates for segment {segment_id}, "
                f"top_score={candidates[0].score:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to publish candidates: {e}")

    def get_metrics(self) -> dict:
        """Get current metrics summary."""
        return self._metrics.get_summary()

    def has_documents(self) -> bool:
        """Check if this room has documents indexed."""
        return self._retriever_manager.has_documents(self.room_id)
