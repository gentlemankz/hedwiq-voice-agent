# CLAUDE.md

## Project Overview

Hedwiq Agent - LiveKit agent for real-time meeting transcription, insight extraction, document reference detection, and agenda tracking.

## Commands

```bash
# Development
python hedwiq_agent.py dev          # Full agent (transcription + insights + doc reference + agenda)
python transcription_agent.py dev   # Transcription only
python document_api.py              # Document upload API (separate process)

# Environment setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Architecture (Phase 4)

```
Audio → VAD/Deepgram STT → lk.transcription topic
                      ↓
              InsightAnalyzer → hedwiq.insight topic
                      ↓
            DocumentReferencer (Phase 3 Pipeline):
                [Pre-filter] → [Hybrid Retrieval] → [LLM Alignment] → [Dedupe]
                   (no LLM)        (~20ms)            (~200ms)
                      ↓
               hedwiq.document_reference topic (confirmed references)
                      ↑
Document API → PersistentDocumentStore → HybridRetriever (BM25 + Embeddings + RRF)
      ↑
Supabase Storage (downloads PDF from frontend uploads)

                      ↓
            AgendaTracker (Phase 4 Pipeline):
                [Transcript Buffer] → [LLM Context Analysis] → [Stability Check]
                   (debounced)           (~500ms)              (hysteresis)
                      ↓
               hedwiq.agenda topic (topic_started, topic_completed, etc.)
```

### Agenda Detection Flow (Phase 4)
```
Transcript → Buffer (3+ segments) → Pure LLM Analysis → Stability Check → Publish Event
                                        ↓
                        No word patterns - LLM analyzes conversation context
                        to determine which agenda topic is being discussed
```

**IMPORTANT**: Agent uses `request_fnc` to set identity prefix to "hedwiq" (required for frontend event filtering). Do NOT use `agent_name` in WorkerOptions - that disables automatic dispatch!

### Document Upload Flow
```
Frontend PreJoin → Supabase Storage → POST /documents/process → Agent processes
                                              ↓
                         Download PDF → Parse → Embed → Store in SQLite
```

## Key Files

| File | Purpose |
|------|---------|
| `hedwiq_agent.py` | Main agent: STT + LLM insights + document reference + agenda |
| `agenda_tracker.py` | Phase 4: Pure LLM topic detection + stability + late joiner sync |
| `hybrid_retriever.py` | BM25 + embedding search with RRF fusion (~20ms) |
| `document_referencer.py` | Phase 3: Pre-filter + Retrieval + LLM alignment + Dedupe |
| `document_api.py` | FastAPI for document upload + Supabase processing |
| `document_processor.py` | PDF parsing + embeddings |
| `persistent_store.py` | SQLite/Redis document storage |
| `supabase_client.py` | Supabase Storage client for downloading PDFs |
| `prompts/agenda_detection.py` | LLM prompts for unified topic detection |
| `prompts/document_reference.py` | LLM alignment prompt for reference validation |
| `schemas/agenda.py` | Agenda event types + detection constants |
| `db/agenda.py` | PostgreSQL client for agenda read/write |

## Configuration

Required in `.env`:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
- `INTERNAL_SERVICE_TOKEN` (for document API)
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (for Supabase Storage integration)

## Key Constants

### Insight Extraction
- `MIN_CONFIDENCE_THRESHOLD = 0.75` - Insight confidence threshold
- `MIN_INSIGHT_WORDS = 8` - Minimum words per insight

### Document Reference (Phase 3)
- `MAX_DOCUMENTS_PER_ROOM = 10` - Document limit
- `MIN_SEGMENT_WORDS = 6` - Pre-filter threshold for retrieval
- `RRF_K = 60` - Reciprocal Rank Fusion constant
- `MIN_ALIGNMENT_CONFIDENCE = 0.7` - LLM alignment confidence threshold
- `ALIGNMENT_TIMEOUT_SECONDS = 2.0` - LLM timeout
- `MAX_CONCURRENT_ALIGNMENTS = 3` - Backpressure limit
- `DEDUPE_TTL_MINUTES = 5` - Reference deduplication TTL

### Agenda Tracking (Phase 4 - Revised v2)
- `STABILITY_CONSECUTIVE_K = 2` - Consecutive predictions needed before transition
- `STABILITY_TIME_THRESHOLD = 4.0` - Seconds of consistent prediction needed
- `SWITCH_CONFIDENCE_THRESHOLD = 0.80` - Minimum LLM confidence for topic switch
- `HYSTERESIS_COOLDOWN = 5.0` - Minimum seconds between topic switches
- `MIN_TIME_ON_TOPIC = 15.0` - **NEW**: Minimum seconds on current topic before allowing transition
- `MIN_ANALYSIS_INTERVAL = 2.0` - Minimum seconds between LLM analyses
- `ANALYSIS_DEBOUNCE_SECONDS = 1.5` - Debounce delay before analysis
- `MIN_SEGMENT_WORDS_FOR_DETECTION = 5` - Minimum words to trigger analysis

**Topic Detection Flow (Revised v2)**:
1. LLM explicitly asked if topic should transition with `should_transition` flag
2. **Critical distinction**: MENTIONS are NOT transitions - must have SUSTAINED DISCUSSION
3. Minimum 15 seconds on current topic before transitions allowed
4. Requires 2 consecutive predictions OR 4 seconds of consistent prediction
5. Recent transcript segments marked with `(RECENT)` to help LLM focus
6. Very high confidence (≥0.90) can bypass stability after min time met

**Transition Requirements**:
- Speaker must be EXPLAINING/ELABORATING on new topic (not just mentioning it)
- Multiple sentences actually ABOUT the new topic content required
- "Today we'll discuss X, Y, Z" = stays on current topic (just listing)
- "Let me explain how X works..." = potential transition (actual discussion)

**Note**: Word-based patterns are DEPRECATED. All detection uses pure LLM context analysis.
