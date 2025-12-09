"""
Document Referencer for Hedwiq Agent - Phase 3 Implementation

Provides real-time document reference detection using hybrid retrieval + LLM alignment.
Integrates with hedwiq_agent.py's VAD/transcript flow.

Phase 3 implements:
- Pre-filter (no LLM) to skip irrelevant segments
- Hybrid retrieval (BM25 + embeddings + RRF)
- Single LLM alignment for validation (NEW in Phase 3)
- Deduplication with TTL
- Timeout/backpressure handling (NEW in Phase 3)
- High-confidence reference publishing

Pipeline:
    [Transcript] → [Pre-filter] → [Hybrid Retrieval] → [LLM Alignment] → [Dedupe] → [Publish]
                     (no LLM)        (~20ms)              (~200ms)

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
import os
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass

from livekit import rtc

from hybrid_retriever import (
    HybridRetriever,
    RoomRetrieverManager,
    RetrievalMetrics,
)
from persistent_store import PersistentDocumentStore
from schemas.documents import (
    RetrievalCandidate,
    DocumentReference,
    BoundingBox,
    DEDUPE_TTL_MINUTES,
)
from prompts.document_reference import (
    format_alignment_prompt,
    MIN_ALIGNMENT_CONFIDENCE,
    ALIGNMENT_TIMEOUT_SECONDS,
    ALIGNMENT_MAX_RETRIES,
)

logger = logging.getLogger("hedwiq-document-referencer")

# LiveKit text stream topics
DOCUMENT_REFERENCE_TOPIC = "hedwiq.document_reference"  # Phase 3: confirmed references
DOCUMENT_CANDIDATE_TOPIC = "hedwiq.document_candidate"  # Phase 2: candidates for preview


@dataclass
class AlignmentResult:
    """Result from LLM alignment step."""
    found: bool
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    evidence_span: Optional[str] = None
    confidence: float = 0.0
    rationale: Optional[str] = None


class DocumentReferencer:
    """
    Real-time document reference detection using hybrid retrieval + LLM alignment.

    This class integrates with hedwiq_agent.py to:
    1. Receive final transcript segments from VAD flow
    2. Apply lightweight pre-filter (no LLM)
    3. Run hybrid retrieval (BM25 + embeddings)
    4. Run single LLM alignment for validation (Phase 3)
    5. Apply deduplication with TTL
    6. Publish confirmed references

    Features:
    - Timeout handling for LLM calls
    - Backpressure: limits concurrent processing
    - Graceful degradation on LLM failures
    - Thread-safe for concurrent transcript processing
    """

    # Backpressure settings
    MAX_QUEUE_SIZE = 50  # Maximum pending segments
    MAX_CONCURRENT_ALIGNMENTS = 3  # Max parallel LLM calls

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

        # Processing queue with backpressure
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._task: Optional[asyncio.Task] = None

        # Concurrency control for LLM alignment
        self._alignment_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_ALIGNMENTS)

        # Deduplication cache: fingerprint -> timestamp
        self._recent_refs: Dict[str, float] = {}
        self._dedupe_lock = asyncio.Lock()

        # Azure OpenAI client for LLM alignment (lazy loaded)
        self._llm_client = None
        self._llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        # Metrics
        self._metrics = RetrievalMetrics(room_id)
        self._alignment_metrics = {
            "total": 0,
            "successful": 0,
            "timeout": 0,
            "error": 0,
            "no_match": 0,
            "low_confidence": 0,  # Rejected due to confidence < MIN_ALIGNMENT_CONFIDENCE
            "total_latency_ms": 0,
        }

        # State
        self._running = False
        self._last_doc_snapshot: Optional[tuple[int, int]] = None  # (count, latest_created_at)

    def _get_llm_client(self):
        """Lazy load Azure OpenAI client for LLM alignment."""
        if self._llm_client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI SDK is required for LLM alignment. "
                    "Install it with: pip install openai"
                )

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-10-01-preview")

            if not api_key or not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                    "environment variables are required for LLM alignment"
                )

            self._llm_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

        return self._llm_client

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
        self._last_doc_snapshot = self._get_doc_snapshot()

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
        self._log_alignment_metrics()

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

        # Refresh index if new documents arrived while agent is running
        await self._maybe_refresh_index()

        # Check if we have any documents for this room (after potential refresh)
        if not self._retriever_manager.has_documents(self.room_id):
            return

        # Queue for async processing with backpressure
        try:
            self._queue.put_nowait({
                "segment_id": segment_id,
                "transcript": transcript,
                "speaker": speaker_identity,
                "duration": duration_seconds,
                "timestamp": time.time(),
            })
        except asyncio.QueueFull:
            logger.warning(
                f"Document reference queue full, dropping segment {segment_id}"
            )

    async def on_document_added(self):
        """
        Called when a document is uploaded to this room.

        Triggers index rebuild.
        """
        self._retriever_manager.rebuild_room_index(self.room_id)
        self._last_doc_snapshot = self._get_doc_snapshot()
        logger.info(f"Rebuilt retrieval index for room {self.room_id} after document upload")

    async def on_document_removed(self):
        """
        Called when a document is removed from this room.

        Triggers index rebuild.
        """
        self._retriever_manager.rebuild_room_index(self.room_id)
        self._last_doc_snapshot = self._get_doc_snapshot()
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
        Process a single transcript segment through the full pipeline.

        Pipeline (Phase 3):
        1. Pre-filter (skip short/filler segments) - no LLM
        2. Hybrid retrieval (BM25 + embeddings + RRF) - ~20ms
        3. LLM alignment (validate reference) - ~200ms
        4. Deduplication
        5. Publish confirmed reference
        """
        transcript = item["transcript"]
        segment_id = item["segment_id"]
        duration = item["duration"]
        speaker = item["speaker"]

        # Get retriever for this room
        retriever = self._retriever_manager.get_retriever(self.room_id)
        if not retriever:
            return

        # Refresh index lazily if uploads changed in another process
        await self._maybe_refresh_index()

        # Step 1: Pre-filter (no LLM)
        prefilter_result = retriever.prefilter_segment(transcript, duration)
        try:
            self._metrics.record_prefilter(
                prefilter_result.should_process,
                prefilter_result.reason
            )
        except Exception as e:
            logger.warning(f"Metrics recording failed (prefilter): {e}")

        if not prefilter_result.should_process:
            return

        # Step 2: Hybrid retrieval
        retrieval_start = time.time()
        candidates = retriever.retrieve(transcript, top_k=3)
        retrieval_latency_ms = (time.time() - retrieval_start) * 1000

        try:
            self._metrics.record_retrieval(retrieval_latency_ms, candidates)
        except Exception as e:
            logger.warning(f"Metrics recording failed (retrieval): {e}")

        if not candidates:
            return

        # Step 3: LLM alignment (Phase 3) with timeout and backpressure
        alignment_result = await self._align_with_llm(transcript, candidates)

        if not alignment_result or not alignment_result.found:
            # Optionally publish candidates for preview even if no confirmed match
            # await self._publish_candidates(segment_id, transcript, speaker, candidates)
            return

        # Step 3b: Enforce confidence threshold (per plan: confidence >= 0.7)
        if alignment_result.confidence < MIN_ALIGNMENT_CONFIDENCE:
            logger.debug(
                f"Low confidence reference skipped: {alignment_result.confidence:.2f} < {MIN_ALIGNMENT_CONFIDENCE}"
            )
            self._alignment_metrics["low_confidence"] = self._alignment_metrics.get("low_confidence", 0) + 1
            return

        # Find the matching candidate for full data
        matching_candidate = next(
            (c for c in candidates if c.segment_id == alignment_result.section_id),
            candidates[0]  # Fallback to top candidate
        )

        # Step 4: Create DocumentReference
        reference = DocumentReference(
            id=f"ref-{uuid.uuid4().hex[:8]}",
            document_id=matching_candidate.document_id,
            section_id=alignment_result.section_id or matching_candidate.segment_id,
            page_number=alignment_result.page_number or matching_candidate.page_number,
            section_title=matching_candidate.section_title,
            matched_text=alignment_result.evidence_span or matching_candidate.content[:100],
            bbox=matching_candidate.bbox,
            # Keep context within schema limit (<=200 chars) to avoid validation errors
            context=(alignment_result.rationale or "Document reference detected")[:200],
            confidence=alignment_result.confidence,
            transcript_ref=segment_id,
        )

        # Step 5: Deduplication
        if await self._is_duplicate(reference):
            logger.debug(f"Duplicate reference skipped: {reference.section_id}")
            return

        # Step 6: Publish confirmed reference
        await self._publish_reference(reference, speaker)

    async def _align_with_llm(
        self,
        transcript: str,
        candidates: List[RetrievalCandidate]
    ) -> Optional[AlignmentResult]:
        """
        Run single LLM alignment to validate reference.

        Features:
        - Timeout handling (2 second default)
        - Retry on transient failures
        - Graceful degradation on errors
        - Concurrency limiting via semaphore

        Args:
            transcript: The speech transcript
            candidates: Retrieved candidates to validate against

        Returns:
            AlignmentResult or None if alignment fails/times out
        """
        self._alignment_metrics["total"] += 1
        start_time = time.time()

        # Acquire semaphore for backpressure
        async with self._alignment_semaphore:
            for attempt in range(ALIGNMENT_MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        self._call_llm_alignment(transcript, candidates),
                        timeout=ALIGNMENT_TIMEOUT_SECONDS
                    )

                    latency_ms = (time.time() - start_time) * 1000
                    self._alignment_metrics["total_latency_ms"] += latency_ms

                    if result and result.found:
                        self._alignment_metrics["successful"] += 1
                        logger.debug(
                            f"LLM alignment found match: {result.section_id} "
                            f"(conf={result.confidence:.2f}, {latency_ms:.0f}ms)"
                        )
                    else:
                        self._alignment_metrics["no_match"] += 1

                    return result

                except asyncio.TimeoutError:
                    self._alignment_metrics["timeout"] += 1
                    logger.warning(
                        f"LLM alignment timeout (attempt {attempt + 1}/{ALIGNMENT_MAX_RETRIES + 1})"
                    )
                    if attempt < ALIGNMENT_MAX_RETRIES:
                        continue
                    return None

                except Exception as e:
                    self._alignment_metrics["error"] += 1
                    logger.error(f"LLM alignment error: {e}")
                    if attempt < ALIGNMENT_MAX_RETRIES:
                        await asyncio.sleep(0.1)  # Brief pause before retry
                        continue
                    return None

        return None

    async def _call_llm_alignment(
        self,
        transcript: str,
        candidates: List[RetrievalCandidate]
    ) -> Optional[AlignmentResult]:
        """
        Make the actual LLM API call for alignment.

        Args:
            transcript: The speech transcript
            candidates: Retrieved candidates

        Returns:
            AlignmentResult parsed from LLM response
        """
        try:
            client = self._get_llm_client()

            # Format prompt
            system_prompt, user_prompt = format_alignment_prompt(transcript, candidates)

            # Call LLM with JSON mode
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )

            # Parse response
            response_text = response.choices[0].message.content
            data = json.loads(response_text)

            return AlignmentResult(
                found=data.get("found", False),
                section_id=data.get("section_id"),
                page_number=data.get("page_number"),
                evidence_span=data.get("evidence_span"),
                confidence=float(data.get("confidence", 0)),
                rationale=data.get("rationale"),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM alignment response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM alignment call failed: {e}")
            raise

    async def _is_duplicate(self, reference: DocumentReference) -> bool:
        """
        Check if this reference is a duplicate within TTL.

        Uses fingerprint of transcript_ref + section_id.
        """
        fingerprint = f"{reference.transcript_ref}:{reference.section_id}"

        async with self._dedupe_lock:
            current_time = time.time()
            ttl_seconds = DEDUPE_TTL_MINUTES * 60

            # Clean old entries
            self._recent_refs = {
                fp: ts for fp, ts in self._recent_refs.items()
                if current_time - ts < ttl_seconds
            }

            # Check for duplicate
            if fingerprint in self._recent_refs:
                return True

            # Add new entry
            self._recent_refs[fingerprint] = current_time
            return False

    async def _publish_reference(self, reference: DocumentReference, speaker: str):
        """
        Publish confirmed document reference via LiveKit text stream.

        Args:
            reference: The confirmed DocumentReference
            speaker: Speaker identity
        """
        try:
            payload = {
                "type": "document_reference",
                "id": reference.id,
                "document_id": reference.document_id,
                "section_id": reference.section_id,
                "page_number": reference.page_number,
                "section_title": reference.section_title,
                "matched_text": reference.matched_text,
                "bbox": reference.bbox.to_dict() if reference.bbox else None,
                "context": reference.context,
                "confidence": reference.confidence,
                "transcript_ref": reference.transcript_ref,
                "timestamp": reference.timestamp,
                "speaker": speaker,
            }

            await self.room.local_participant.send_text(
                json.dumps(payload),
                topic=DOCUMENT_REFERENCE_TOPIC,
                attributes={
                    "document_id": reference.document_id,
                    "section_id": reference.section_id,
                    "page_number": str(reference.page_number),
                    "confidence": str(round(reference.confidence, 2)),
                },
            )

            logger.info(
                f"Published reference: {reference.context[:50]}... "
                f"(conf={reference.confidence:.2f}, page={reference.page_number})"
            )

        except Exception as e:
            logger.error(f"Failed to publish reference: {e}")

    def _get_doc_snapshot(self) -> Optional[tuple[int, int]]:
        """
        Return a lightweight snapshot (count, newest created_at) to detect changes.
        """
        try:
            docs = self._store.get_documents_for_room(self.room_id)
            if not docs:
                return (0, 0)
            latest = max(d.created_at for d in docs)
            return (len(docs), latest)
        except Exception as e:
            logger.warning(f"Failed to compute document snapshot: {e}")
            return None

    async def _maybe_refresh_index(self):
        """
        Rebuild the retrieval index if the underlying document set changed.

        Handles uploads performed in a separate process (document_api).
        """
        snapshot = self._get_doc_snapshot()
        if snapshot is None:
            return

        if self._last_doc_snapshot is None or snapshot != self._last_doc_snapshot:
            self._retriever_manager.rebuild_room_index(self.room_id)
            self._last_doc_snapshot = snapshot
            logger.info(f"Rebuilt retrieval index for room {self.room_id} (documents changed)")

    async def _publish_candidates(
        self,
        segment_id: str,
        transcript: str,
        speaker: str,
        candidates: List[RetrievalCandidate]
    ):
        """
        Publish retrieval candidates for frontend preview (optional).

        This can be used to show potential matches even before LLM confirmation.
        Kept from Phase 2 for backwards compatibility.

        Args:
            segment_id: Transcript segment ID
            transcript: Original transcript text
            speaker: Speaker identity
            candidates: Retrieved candidates
        """
        try:
            payload = {
                "type": "retrieval_candidates",
                "segment_id": segment_id,
                "transcript": transcript[:200],
                "speaker": speaker,
                "timestamp": int(time.time() * 1000),
                "candidates": [
                    {
                        "segment_id": c.segment_id,
                        "document_id": c.document_id,
                        "page_number": c.page_number,
                        "section_title": c.section_title,
                        "content": c.content[:300],
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

            logger.debug(
                f"Published {len(candidates)} candidates for segment {segment_id}"
            )

        except Exception as e:
            logger.error(f"Failed to publish candidates: {e}")

    def _log_alignment_metrics(self):
        """Log alignment metrics summary."""
        m = self._alignment_metrics
        if m["total"] == 0:
            return

        avg_latency = m["total_latency_ms"] / m["total"] if m["total"] > 0 else 0
        success_rate = m["successful"] / m["total"] if m["total"] > 0 else 0

        logger.info(
            f"[{self.room_id}] Alignment metrics: "
            f"total={m['total']}, "
            f"success={m['successful']} ({success_rate:.0%}), "
            f"no_match={m['no_match']}, "
            f"low_conf={m['low_confidence']}, "
            f"timeout={m['timeout']}, "
            f"error={m['error']}, "
            f"avg_latency={avg_latency:.0f}ms"
        )

    def get_metrics(self) -> dict:
        """Get current metrics summary."""
        retrieval_metrics = self._metrics.get_summary()
        retrieval_metrics["alignment"] = {
            **self._alignment_metrics,
            "avg_latency_ms": (
                self._alignment_metrics["total_latency_ms"] / self._alignment_metrics["total"]
                if self._alignment_metrics["total"] > 0 else 0
            ),
        }
        return retrieval_metrics

    def has_documents(self) -> bool:
        """Check if this room has documents indexed."""
        return self._retriever_manager.has_documents(self.room_id)
