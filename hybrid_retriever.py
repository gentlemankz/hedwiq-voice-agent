"""
Hybrid Retriever for Hedwiq Agent - Phase 2 Implementation

Provides hybrid retrieval combining BM25 (lexical) and embeddings (semantic)
with Reciprocal Rank Fusion for document reference detection.

Key features:
- BM25 lexical search (~5ms) for exact keyword matching
- Embedding similarity search (~10ms) for semantic matching
- Reciprocal Rank Fusion (~1ms) to combine results
- Lightweight pre-filter (no LLM) to skip irrelevant segments
- Room-scoped retriever management via RoomRetrieverManager
- ~20ms total latency for retrieval step

Usage:
    # Get room-scoped retriever manager (singleton)
    manager = RoomRetrieverManager.get_instance(document_store)

    # Build/rebuild index for a room (called after document upload)
    await manager.rebuild_room_index(room_id)

    # Get retriever for a room
    retriever = manager.get_retriever(room_id)

    # Check if segment should be processed
    if retriever and retriever.prefilter_segment(transcript, duration).should_process:
        candidates = retriever.retrieve(transcript, top_k=3)
        # Pass candidates to LLM alignment step
"""

import re
import logging
import os
import time
import threading
from typing import List, Optional, Dict, Set, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from schemas.documents import (
    BoundingBox,
    RetrievalCandidate,
    MIN_SEGMENT_WORDS,
    MIN_SEGMENT_DURATION,
    RRF_K,
)

if TYPE_CHECKING:
    from persistent_store import PersistentDocumentStore

logger = logging.getLogger("hedwiq-hybrid-retriever")


# Stop phrases for pre-filter (greetings, fillers, etc.)
# NOTE: These are only applied to SHORT segments (< 12 words) to avoid
# filtering out content-rich segments that happen to start with common words.
STOP_PHRASES: Set[str] = {
    # Greetings
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "bye", "goodbye", "see you", "see you later", "take care",
    # Pleasantries
    "thanks", "thank you", "you're welcome", "no problem", "no worries",
    "please", "sorry", "excuse me", "pardon",
    # Fillers (removed "so" - too common in substantive speech)
    "okay", "ok", "alright", "sure", "yeah", "yes", "no", "uh", "um",
    "hmm", "huh", "well", "anyway", "anyways", "like",
    # Thinking phrases
    "let me think", "i think", "you know", "i mean", "basically",
    "actually", "honestly", "to be honest", "in my opinion",
    # Technical/meeting phrases
    "can you hear me", "can everyone hear me", "is everyone here",
    "let's get started", "let's begin", "shall we start",
    "one moment", "hold on", "wait a second", "just a minute",
    # Short responses
    "got it", "i see", "i understand", "makes sense", "right",
    "exactly", "correct", "absolutely", "definitely",
}

# Threshold for applying stop phrase filter (only for short segments)
STOP_PHRASE_MAX_WORDS = 12

# BM25 stopwords for tokenization
BM25_STOPWORDS: Set[str] = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
    'neither', 'not', 'only', 'own', 'same', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'any', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
    'its', 'they', 'them', 'their', 'what', 'which', 'who',
    'this', 'that', 'these', 'those', 'am',
}


@dataclass
class SegmentPrefilterResult:
    """Result of pre-filtering a transcript segment."""
    should_process: bool
    reason: Optional[str] = None


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) and embeddings (semantic).

    This class implements the retrieval-first architecture from Phase 2:
    - Higher recall via dual retrieval methods
    - Lower latency (~20ms vs ~300ms for 2 LLM calls)
    - No LLM tokens for retrieval step

    The retrieval pipeline:
    1. Pre-filter: Skip short/filler segments (no LLM)
    2. BM25 Search: Top-10 lexical matches (~5ms)
    3. Embedding Search: Top-10 semantic matches (~10ms)
    4. RRF Fusion: Combine rankings for top-3 final candidates (~1ms)
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        min_score_threshold: float = 0.02,  # Minimum RRF score to return candidate
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_model: Azure OpenAI embedding model deployment name
            min_score_threshold: Minimum RRF score to return a candidate
        """
        self.embedding_model = embedding_model
        self.min_score_threshold = min_score_threshold

        # BM25 index (built when segments are loaded)
        self._bm25_index: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

        # Segment data
        self._segments: List[dict] = []
        self._segment_embeddings: Optional[np.ndarray] = None

        # Azure OpenAI client (lazy loaded)
        self._openai_client = None

        # Previous segments for overlap detection (pre-filter)
        self._prev_segments: List[str] = []

    def _get_openai_client(self):
        """Lazy load Azure OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI SDK is required for embeddings. "
                    "Install it with: pip install openai"
                )

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")

            if not api_key or not endpoint:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                    "environment variables are required for embeddings"
                )

            self._openai_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

        return self._openai_client

    # =========================================================================
    # Index Building
    # =========================================================================

    def build_index(
        self,
        segments: List[dict],
        embeddings: Optional[np.ndarray] = None
    ):
        """
        Build BM25 and embedding indices from segments.

        This should be called whenever documents are added/removed.

        Args:
            segments: List of segment dictionaries with 'id', 'content', etc.
            embeddings: Pre-computed embeddings (optional, will compute if None)
        """
        if not segments:
            logger.warning("No segments provided, clearing indices")
            self._clear_index()
            return

        self._segments = segments

        # Build BM25 index
        self._tokenized_corpus = [
            self._tokenize(s.get("content", ""))
            for s in segments
        ]
        self._bm25_index = BM25Okapi(self._tokenized_corpus)

        # Store or compute embeddings
        if embeddings is not None and len(embeddings) == len(segments):
            self._segment_embeddings = embeddings
        else:
            logger.info("Computing embeddings for segments...")
            self._segment_embeddings = self._compute_embeddings(
                [s.get("content", "") for s in segments]
            )

        logger.info(
            f"Built hybrid index with {len(segments)} segments, "
            f"embedding dim: {self._segment_embeddings.shape[1] if self._segment_embeddings is not None else 0}"
        )

    def _clear_index(self):
        """Clear all indices."""
        self._bm25_index = None
        self._tokenized_corpus = []
        self._segments = []
        self._segment_embeddings = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Performs:
        - Lowercasing
        - Punctuation removal
        - Stopword removal
        - Minimum token length filtering
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split and filter
        tokens = text.split()
        return [
            t for t in tokens
            if t not in BM25_STOPWORDS and len(t) > 2
        ]

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts using Azure OpenAI.

        Processes in batches to avoid rate limits.
        """
        if not texts:
            return np.array([])

        client = self._get_openai_client()
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to compute embeddings for batch {i}: {e}")
                raise

        return np.array(all_embeddings)

    def _compute_single_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text (for queries)."""
        client = self._get_openai_client()

        try:
            response = client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to compute query embedding: {e}")
            raise

    # =========================================================================
    # Pre-filter Logic (No LLM)
    # =========================================================================

    def prefilter_segment(
        self,
        transcript: str,
        duration: float = 2.0,
        check_overlap: bool = True
    ) -> SegmentPrefilterResult:
        """
        Lightweight pre-filter to skip irrelevant segments without LLM.

        This filter runs BEFORE retrieval to save compute on:
        - Very short segments (< MIN_SEGMENT_WORDS words or < MIN_SEGMENT_DURATION s)
        - Greetings, fillers, pleasantries (only for short segments)
        - Repeated/similar content (chit-chat)

        Args:
            transcript: The transcript text to filter
            duration: Duration of the speech segment in seconds
            check_overlap: Whether to check for overlap with previous segments

        Returns:
            SegmentPrefilterResult indicating if segment should be processed
        """
        # Check 1: Length (word count)
        words = transcript.split()
        word_count = len(words)
        if word_count < MIN_SEGMENT_WORDS:
            return SegmentPrefilterResult(
                should_process=False,
                reason=f"Too short ({word_count} words, min {MIN_SEGMENT_WORDS})"
            )

        # Check 2: Duration
        if duration < MIN_SEGMENT_DURATION:
            return SegmentPrefilterResult(
                should_process=False,
                reason=f"Too brief ({duration:.1f}s, min {MIN_SEGMENT_DURATION}s)"
            )

        # Check 3: Stop phrases (greetings, fillers) - ONLY for short segments
        # Longer segments starting with common words like "so" may still have content
        if word_count < STOP_PHRASE_MAX_WORDS:
            transcript_lower = transcript.lower().strip()
            for phrase in STOP_PHRASES:
                if transcript_lower == phrase or transcript_lower.startswith(phrase + " "):
                    return SegmentPrefilterResult(
                        should_process=False,
                        reason=f"Stop phrase detected: '{phrase}'"
                    )

        # Check 4: High overlap with recent segments (dedupe chit-chat)
        if check_overlap and self._prev_segments:
            for prev in self._prev_segments:
                similarity = self._text_similarity(transcript, prev)
                if similarity > 0.8:
                    return SegmentPrefilterResult(
                        should_process=False,
                        reason=f"High overlap with recent segment ({similarity:.0%})"
                    )

        # Update previous segments buffer
        if check_overlap:
            self._prev_segments = [transcript] + self._prev_segments[:1]

        return SegmentPrefilterResult(
            should_process=True,
            reason=None
        )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word overlap similarity (Jaccard index).

        This is fast (~0.1ms) and good enough for detecting repetitive speech.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # Hybrid Retrieval
    # =========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 10,
        embedding_candidates: int = 10
    ) -> List[RetrievalCandidate]:
        """
        Retrieve top-k candidates using hybrid BM25 + embedding search.

        Pipeline:
        1. BM25 lexical search -> top-N candidates
        2. Embedding similarity search -> top-N candidates
        3. Reciprocal Rank Fusion -> top-k final candidates

        Returns empty list if:
        - No segments indexed
        - No candidates above threshold

        Args:
            query: The transcript text to search for
            top_k: Number of final candidates to return
            bm25_candidates: Number of candidates from BM25 search
            embedding_candidates: Number of candidates from embedding search

        Returns:
            List of RetrievalCandidate objects, sorted by RRF score
        """
        if not self._segments or self._bm25_index is None:
            logger.debug("No segments indexed, returning empty results")
            return []

        if self._segment_embeddings is None or len(self._segment_embeddings) == 0:
            logger.debug("No embeddings indexed, returning empty results")
            return []

        # Step 1: BM25 lexical search
        bm25_results = self._bm25_search(query, top_n=bm25_candidates)

        # Step 2: Embedding similarity search
        embedding_results = self._embedding_search(query, top_n=embedding_candidates)

        # Step 3: Reciprocal Rank Fusion
        candidates = self._reciprocal_rank_fusion(
            bm25_results,
            embedding_results,
            top_k=top_k
        )

        logger.debug(
            f"Retrieved {len(candidates)} candidates for query: '{query[:50]}...'"
        )

        return candidates

    def _bm25_search(
        self,
        query: str,
        top_n: int = 10
    ) -> List[tuple[int, float]]:
        """
        Perform BM25 lexical search.

        Args:
            query: Query text
            top_n: Number of top results to return

        Returns:
            List of (segment_index, bm25_score) tuples
        """
        if self._bm25_index is None:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        # Get BM25 scores
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-N indices
        top_indices = np.argsort(scores)[-top_n:][::-1]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _embedding_search(
        self,
        query: str,
        top_n: int = 10
    ) -> List[tuple[int, float]]:
        """
        Perform embedding similarity search.

        Args:
            query: Query text
            top_n: Number of top results to return

        Returns:
            List of (segment_index, similarity_score) tuples
        """
        if self._segment_embeddings is None or len(self._segment_embeddings) == 0:
            return []

        # Compute query embedding
        try:
            query_embedding = self._compute_single_embedding(query)
        except Exception as e:
            logger.error(f"Failed to compute query embedding: {e}")
            return []

        # Cosine similarity (embeddings are normalized by Azure OpenAI)
        similarities = np.dot(self._segment_embeddings, query_embedding)

        # Get top-N indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[tuple[int, float]],
        embedding_results: List[tuple[int, float]],
        top_k: int = 3
    ) -> List[RetrievalCandidate]:
        """
        Combine BM25 and embedding results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) for each ranking

        This gives more weight to items that appear in both rankings
        and near the top of each ranking.

        Args:
            bm25_results: Results from BM25 search
            embedding_results: Results from embedding search
            top_k: Number of final candidates to return

        Returns:
            List of RetrievalCandidate objects
        """
        rrf_scores: Dict[int, float] = {}

        # Add BM25 rankings
        for rank, (idx, _score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # Add embedding rankings
        for rank, (idx, _score) in enumerate(embedding_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # Sort by RRF score
        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        # Build candidates
        candidates = []
        for idx in sorted_indices[:top_k]:
            score = rrf_scores[idx]

            # Skip if below threshold
            if score < self.min_score_threshold:
                continue

            segment = self._segments[idx]

            # Parse bbox if present
            bbox = None
            if segment.get("bbox"):
                bbox_data = segment["bbox"]
                if isinstance(bbox_data, dict):
                    bbox = BoundingBox(
                        x0=bbox_data.get("x0", 0),
                        y0=bbox_data.get("y0", 0),
                        x1=bbox_data.get("x1", 0),
                        y1=bbox_data.get("y1", 0)
                    )

            candidate = RetrievalCandidate(
                segment_id=segment.get("id", f"seg-{idx}"),
                document_id=segment.get("document_id", "unknown"),
                page_number=segment.get("page_number", 1),
                section_title=segment.get("section_title"),
                content=segment.get("content", ""),
                score=score,
                bbox=bbox
            )
            candidates.append(candidate)

        return candidates

    # =========================================================================
    # Index Status
    # =========================================================================

    def is_ready(self) -> bool:
        """Check if the retriever has segments indexed and ready for search."""
        return (
            self._bm25_index is not None
            and len(self._segments) > 0
            and self._segment_embeddings is not None
            and len(self._segment_embeddings) > 0
        )

    def segment_count(self) -> int:
        """Return the number of indexed segments."""
        return len(self._segments)

    def clear_previous_segments(self):
        """Clear the previous segments buffer used for overlap detection."""
        self._prev_segments = []

    def get_metrics(self) -> Dict[str, any]:
        """Get retrieval metrics for monitoring."""
        return {
            "segment_count": len(self._segments),
            "embedding_dim": self._segment_embeddings.shape[1] if self._segment_embeddings is not None and len(self._segment_embeddings) > 0 else 0,
            "is_ready": self.is_ready(),
            "bm25_indexed": self._bm25_index is not None,
        }


# =============================================================================
# Room Retriever Manager
# =============================================================================

class RoomRetrieverManager:
    """
    Manages room-scoped HybridRetriever instances.

    This class:
    - Creates and caches retrievers per room
    - Rebuilds indices when documents are added/removed
    - Uses precomputed embeddings from PersistentDocumentStore
    - Thread-safe for concurrent access

    Usage:
        manager = RoomRetrieverManager(document_store)
        manager.rebuild_room_index(room_id)  # After document upload
        retriever = manager.get_retriever(room_id)
    """

    _instance: Optional["RoomRetrieverManager"] = None
    _lock = threading.Lock()

    def __init__(self, document_store: "PersistentDocumentStore"):
        """
        Initialize manager with document store.

        Args:
            document_store: PersistentDocumentStore instance for accessing documents
        """
        self._store = document_store
        self._retrievers: Dict[str, HybridRetriever] = {}
        self._retriever_lock = threading.Lock()

        # Metrics
        self._rebuild_count = 0
        self._total_rebuild_time_ms = 0

    @classmethod
    def get_instance(
        cls,
        document_store: Optional["PersistentDocumentStore"] = None
    ) -> "RoomRetrieverManager":
        """
        Get singleton instance of RoomRetrieverManager.

        Args:
            document_store: Required on first call, ignored on subsequent calls

        Returns:
            RoomRetrieverManager singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                if document_store is None:
                    raise ValueError(
                        "document_store is required on first call to get_instance()"
                    )
                cls._instance = cls(document_store)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def get_retriever(self, room_id: str) -> Optional[HybridRetriever]:
        """
        Get the HybridRetriever for a room.

        Returns None if no documents are indexed for the room.

        Args:
            room_id: LiveKit room ID

        Returns:
            HybridRetriever instance or None
        """
        with self._retriever_lock:
            retriever = self._retrievers.get(room_id)
            if retriever and retriever.is_ready():
                return retriever
            return None

    def rebuild_room_index(self, room_id: str) -> bool:
        """
        Rebuild the retrieval index for a room from stored documents.

        This should be called:
        - After document upload
        - After document deletion
        - When agent joins a room with existing documents

        Uses precomputed embeddings from PersistentDocumentStore.

        Args:
            room_id: LiveKit room ID

        Returns:
            True if index was built successfully, False if no documents
        """
        start_time = time.time()

        try:
            # Get segments and embeddings from store
            segments = self._store.get_all_segments_for_room(room_id)
            embeddings = self._store.get_embeddings_for_room(room_id)

            if not segments:
                logger.info(f"No segments for room {room_id}, clearing retriever")
                with self._retriever_lock:
                    if room_id in self._retrievers:
                        del self._retrievers[room_id]
                return False

            # Validate embeddings match segments
            if embeddings is None or len(embeddings) != len(segments):
                logger.warning(
                    f"Embedding count mismatch for room {room_id}: "
                    f"{len(embeddings) if embeddings is not None else 0} embeddings, "
                    f"{len(segments)} segments. Will recompute embeddings."
                )
                embeddings = None

            # Create or update retriever
            with self._retriever_lock:
                if room_id not in self._retrievers:
                    self._retrievers[room_id] = HybridRetriever()

                retriever = self._retrievers[room_id]
                retriever.build_index(segments, embeddings)

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._rebuild_count += 1
            self._total_rebuild_time_ms += elapsed_ms

            logger.info(
                f"Built retrieval index for room {room_id}: "
                f"{len(segments)} segments in {elapsed_ms:.1f}ms"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild index for room {room_id}: {e}")
            return False

    def remove_room(self, room_id: str):
        """
        Remove retriever for a room (when room ends or documents cleared).

        Args:
            room_id: LiveKit room ID
        """
        with self._retriever_lock:
            if room_id in self._retrievers:
                del self._retrievers[room_id]
                logger.info(f"Removed retriever for room {room_id}")

    def has_documents(self, room_id: str) -> bool:
        """
        Check if a room has any documents indexed.

        Args:
            room_id: LiveKit room ID

        Returns:
            True if room has indexed documents
        """
        retriever = self.get_retriever(room_id)
        return retriever is not None and retriever.is_ready()

    def get_metrics(self) -> Dict[str, any]:
        """Get manager-level metrics."""
        with self._retriever_lock:
            room_metrics = {}
            for room_id, retriever in self._retrievers.items():
                room_metrics[room_id] = retriever.get_metrics()

            return {
                "active_rooms": len(self._retrievers),
                "total_rebuilds": self._rebuild_count,
                "avg_rebuild_time_ms": (
                    self._total_rebuild_time_ms / self._rebuild_count
                    if self._rebuild_count > 0 else 0
                ),
                "rooms": room_metrics,
            }


# =============================================================================
# Retrieval Metrics Logger
# =============================================================================

class RetrievalMetrics:
    """
    Tracks and logs retrieval performance metrics.

    Provides visibility into:
    - Pre-filter pass/skip rates
    - Retrieval latency
    - Candidate counts and scores
    - BM25 vs embedding contribution
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self._prefilter_total = 0
        self._prefilter_passed = 0
        self._retrieval_total = 0
        self._retrieval_with_candidates = 0
        self._total_latency_ms = 0
        self._candidate_scores: List[float] = []

    def record_prefilter(self, passed: bool, reason: Optional[str] = None):
        """Record a pre-filter decision."""
        self._prefilter_total += 1
        if passed:
            self._prefilter_passed += 1
        else:
            logger.debug(f"[{self.room_id}] Pre-filter skip: {reason}")

    def record_retrieval(
        self,
        latency_ms: float,
        candidates: List[RetrievalCandidate]
    ):
        """Record a retrieval operation."""
        self._retrieval_total += 1
        self._total_latency_ms += latency_ms

        if candidates:
            self._retrieval_with_candidates += 1
            self._candidate_scores.extend([c.score for c in candidates])

        top_score = candidates[0].score if candidates else 0.0
        logger.debug(
            f"[{self.room_id}] Retrieval: {len(candidates)} candidates "
            f"in {latency_ms:.1f}ms, top_score={top_score:.3f}"
        )

    def get_summary(self) -> Dict[str, any]:
        """Get metrics summary."""
        return {
            "room_id": self.room_id,
            "prefilter": {
                "total": self._prefilter_total,
                "passed": self._prefilter_passed,
                "pass_rate": (
                    self._prefilter_passed / self._prefilter_total
                    if self._prefilter_total > 0 else 0
                ),
            },
            "retrieval": {
                "total": self._retrieval_total,
                "with_candidates": self._retrieval_with_candidates,
                "hit_rate": (
                    self._retrieval_with_candidates / self._retrieval_total
                    if self._retrieval_total > 0 else 0
                ),
                "avg_latency_ms": (
                    self._total_latency_ms / self._retrieval_total
                    if self._retrieval_total > 0 else 0
                ),
                "avg_top_score": (
                    sum(self._candidate_scores) / len(self._candidate_scores)
                    if self._candidate_scores else 0
                ),
            },
        }

    def log_summary(self):
        """Log metrics summary."""
        summary = self.get_summary()
        logger.info(
            f"[{self.room_id}] Retrieval metrics: "
            f"prefilter_pass={summary['prefilter']['pass_rate']:.0%}, "
            f"retrieval_hit={summary['retrieval']['hit_rate']:.0%}, "
            f"avg_latency={summary['retrieval']['avg_latency_ms']:.1f}ms"
        )
