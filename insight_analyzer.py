"""
Insight analysis helpers for Hedwiq Agent.

Extracted from hedwiq_agent to keep the main orchestration lean.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from livekit.agents import llm

from schemas.insights import Insight, InsightType
from prompts.insight_extraction import (
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    INSIGHT_EXTRACTION_USER_TEMPLATE,
)

# Constants
MIN_ANALYSIS_INTERVAL = 5.0
MIN_SEGMENTS_FOR_ANALYSIS = 3
ANALYSIS_DELAY = 3.0
MAX_TRANSCRIPT_BUFFER = 30
MIN_CONFIDENCE_THRESHOLD = 0.75
MIN_INSIGHT_WORDS = 8
SEMANTIC_SIMILARITY_THRESHOLD = 0.5

logger = logging.getLogger("hedwiq-agent")


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
    """

    room: any
    llm: any
    transcript_buffer: List[TranscriptEntry] = field(default_factory=list)
    pending_segments: List[TranscriptEntry] = field(default_factory=list)
    published_insights: set = field(default_factory=set)
    published_contents: List[str] = field(default_factory=list)
    recent_insight_summaries: List[dict] = field(default_factory=list)
    last_analysis_time: float = 0
    analysis_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    schedule_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduled_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        self.analysis_lock = asyncio.Lock()
        self.schedule_lock = asyncio.Lock()

    async def add_transcript(self, entry: TranscriptEntry):
        """Add a transcript entry and schedule analysis."""
        if not entry.is_final:
            return

        async with self.schedule_lock:
            self.pending_segments.append(entry)

            now = time.time()
            time_since_last = now - self.last_analysis_time
            enough_segments = len(self.pending_segments) >= MIN_SEGMENTS_FOR_ANALYSIS
            enough_time = time_since_last >= MIN_ANALYSIS_INTERVAL

            should_schedule = enough_segments or (self.pending_segments and enough_time)

            if should_schedule and (self.scheduled_task is None or self.scheduled_task.done()):
                self.scheduled_task = asyncio.create_task(self._delayed_analysis())

    async def _delayed_analysis(self):
        await asyncio.sleep(ANALYSIS_DELAY)
        await self._run_analysis()

    async def _run_analysis(self):
        async with self.analysis_lock:
            async with self.schedule_lock:
                if not self.pending_segments:
                    return
                segments_to_analyze = self.pending_segments.copy()
                self.pending_segments.clear()

            self.transcript_buffer.extend(segments_to_analyze)
            if len(self.transcript_buffer) > MAX_TRANSCRIPT_BUFFER:
                self.transcript_buffer = self.transcript_buffer[-MAX_TRANSCRIPT_BUFFER:]

            self.last_analysis_time = time.time()
            await self._extract_insights()

    async def _extract_insights(self):
        if not self.transcript_buffer:
            return

        transcript_text, speaker_map = self._build_transcript_context()
        previous_insights = self._build_previous_insights_summary()

        if not transcript_text.strip():
            return

        user_prompt = INSIGHT_EXTRACTION_USER_TEMPLATE.format(
            transcript=transcript_text,
            speaker_map=json.dumps(speaker_map, indent=2),
            previous_insights=previous_insights,
        )

        for attempt in range(2):
            try:
                chat_ctx = llm.ChatContext()
                chat_ctx.add_message(role="system", content=INSIGHT_EXTRACTION_SYSTEM_PROMPT)
                chat_ctx.add_message(role="user", content=user_prompt)

                response_text = ""
                stream = self.llm.chat(chat_ctx=chat_ctx)
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        response_text += chunk.delta.content

                insights = self._parse_insights(response_text, speaker_map)

                if insights is None and attempt == 0:
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
        merged = self._merge_speaker_turns(self.transcript_buffer[-15:])
        lines = []
        speaker_map: Dict[str, str] = {}
        for turn in merged:
            speaker_map[turn["identity"]] = turn["name"]
            lines.append(f"[{turn['identity']}]: {turn['text']}")
        return "\n".join(lines), speaker_map

    def _merge_speaker_turns(self, entries: List[TranscriptEntry]) -> List[dict]:
        if not entries:
            return []
        merged = []
        for entry in entries:
            if merged and merged[-1]["identity"] == entry.speaker_identity:
                merged[-1]["text"] += " " + entry.text
                merged[-1]["segment_id"] = entry.segment_id
            else:
                merged.append({
                    "identity": entry.speaker_identity,
                    "name": entry.speaker_name,
                    "text": entry.text,
                    "segment_id": entry.segment_id,
                })
        return merged

    def _build_previous_insights_summary(self) -> str:
        if not self.recent_insight_summaries:
            return "None yet."
        lines = []
        for insight in self.recent_insight_summaries[-10:]:
            lines.append(f"- [{insight['type']}] {insight['content'][:60]}...")
        return "\n".join(lines)

    def _content_fingerprint(self, insight_type: str, content: str, speaker: str) -> str:
        normalized = f"{insight_type}:{content.lower().strip()}:{speaker or 'unknown'}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _is_semantically_similar(self, new_content: str) -> bool:
        new_words = set(new_content.lower().split())
        for existing in self.published_contents[-50:]:
            existing_words = set(existing.lower().split())
            if not new_words or not existing_words:
                continue
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and (intersection / union) > SEMANTIC_SIMILARITY_THRESHOLD:
                return True
        return False

    def _parse_insights(self, response: str, speaker_map: dict) -> Optional[List[Insight]]:
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            if not cleaned or cleaned == "[]":
                return []

            data = json.loads(cleaned)

            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data)}")
                return None

            insights: List[Insight] = []
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
            return None

    def _validate_and_create_insight(self, item: dict, speaker_map: dict) -> Optional[Insight]:
        insight_type = item.get("type", "").lower()
        if insight_type not in [t.value for t in InsightType]:
            return None

        confidence = float(item.get("confidence", 0.8))
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return None

        content = item.get("content", "").strip()
        word_count = len(content.split())
        if word_count < MIN_INSIGHT_WORDS:
            return None

        speaker_from_llm = item.get("speaker", "")
        fingerprint = self._content_fingerprint(insight_type, content, speaker_from_llm)
        if fingerprint in self.published_insights:
            return None

        if self._is_semantically_similar(content):
            return None

        speaker_identity = speaker_from_llm
        speaker_name = speaker_from_llm

        if speaker_from_llm in speaker_map:
            speaker_name = speaker_map[speaker_from_llm]
        else:
            for identity, name in speaker_map.items():
                if name.lower() == speaker_from_llm.lower():
                    speaker_identity = identity
                    speaker_name = name
                    break
            else:
                if self.transcript_buffer:
                    speaker_identity = self.transcript_buffer[-1].speaker_identity
                    speaker_name = self.transcript_buffer[-1].speaker_name

        transcript_ref = None
        for entry in reversed(self.transcript_buffer):
            if entry.speaker_identity == speaker_identity:
                transcript_ref = entry.segment_id
                break
        if transcript_ref is None and self.transcript_buffer:
            transcript_ref = self.transcript_buffer[-1].segment_id

        insight = Insight(
            type=InsightType(insight_type),
            content=content,
            speaker=speaker_identity,
            speaker_name=speaker_name,
            confidence=confidence,
            transcript_ref=transcript_ref,
            timestamp=int(time.time() * 1000),
        )

        self.published_insights.add(fingerprint)
        self.published_contents.append(content)
        if len(self.published_contents) > 100:
            self.published_contents = self.published_contents[-100:]

        self.recent_insight_summaries.append({"type": insight_type, "content": content})
        if len(self.recent_insight_summaries) > 20:
            self.recent_insight_summaries = self.recent_insight_summaries[-20:]

        return insight

    async def _publish_insight(self, insight: Insight):
        try:
            insight_data = {
                "id": str(uuid.uuid4()),
                "type": insight.type,
                "content": insight.content,
                "speaker": insight.speaker,
                "speakerName": insight.speaker_name,
                "confidence": insight.confidence,
                "transcriptRef": insight.transcript_ref,
                "timestamp": insight.timestamp,
            }

            await self.room.local_participant.send_text(
                json.dumps(insight_data),
                topic="hedwiq.insight",
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

