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
            AgendaTracker (Phase 4 - Trust-Based LLM):
                [Full Transcript Buffer] → [LLM Full Context Analysis] → [Execute Transition]
                     (all entries)              (~500ms)                   (trust LLM)
                      ↓
               hedwiq.agenda topic (topic_started, topic_completed, etc.)
```

### Agenda Detection Flow (Phase 4 - Trust-Based LLM)
```
Transcript → Full Buffer → LLM Full Context Analysis → Execute Transition
                                    ↓
                    Give LLM FULL conversation history
                    Ask: "Has speaker MOVED ON to new topic?"
                    Trust LLM decision - no artificial constraints
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
| `agenda_tracker.py` | Phase 4: Trust-based LLM topic detection + late joiner sync |
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

### Agenda Tracking (Phase 4 - Trust-Based LLM Architecture)

**Philosophy**: Modern LLMs are intelligent enough to understand conversation context.
Instead of adding "magic constants" that second-guess the LLM, we give it full
conversation history and trust its judgment.

**Minimal Constants (for performance only)**:
- `MIN_ANALYSIS_INTERVAL = 1.5` - Rate limiting between LLM analyses
- `ANALYSIS_DEBOUNCE_SECONDS = 1.0` - Debounce delay to batch transcript segments
- `MAX_TRANSCRIPT_BUFFER = 100` - Maximum transcript entries (soft limit for context)
- `MIN_SEGMENT_WORDS_FOR_DETECTION = 3` - Skip very short utterances ("um", "uh")

**REMOVED (no longer used)**:
- ~~STABILITY_CONSECUTIVE_K~~ - No stability checks
- ~~SWITCH_CONFIDENCE_THRESHOLD~~ - No confidence thresholds
- ~~HYSTERESIS_COOLDOWN~~ - No artificial cooldowns
- ~~MIN_TIME_ON_TOPIC~~ - No minimum time constraints

**Topic Detection Flow (Trust-Based)**:
1. Give LLM FULL conversation transcript (not just recent segments)
2. Ask: "Has speaker intentionally MOVED ON to discussing a new topic?"
3. Key distinction: MENTIONS ≠ TRANSITIONS (listing topics vs discussing them)
4. Trust LLM's decision - if it says transition, we transition
5. No confidence thresholds, no stability checks, no magic constants

**Transition Guidance (in LLM prompt)**:
- Speaker must be EXPLAINING/ELABORATING on new topic (not just mentioning it)
- "Today we'll discuss X, Y, Z" = stays on current topic (just listing)
- "Let me explain how X works..." = potential transition (actual discussion)
- LLM sees full context and makes intelligent decision
