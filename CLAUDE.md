# CLAUDE.md

## Project Overview

Hedwiq Agent - LiveKit agent for real-time meeting transcription, insight extraction, and document reference detection.

## Commands

```bash
# Development
python hedwiq_agent.py dev          # Full agent (transcription + insights + doc reference)
python transcription_agent.py dev   # Transcription only
python document_api.py              # Document upload API (separate process)

# Environment setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Architecture (Phase 3)

```
Audio → Deepgram STTv2 (WebSocket) → lk.transcription topic
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
```

### Document Upload Flow
```
Frontend PreJoin → Supabase Storage → POST /documents/process → Agent processes
                                              ↓
                         Download PDF → Parse → Embed → Store in SQLite
```

## Key Files

| File | Purpose |
|------|---------|
| `hedwiq_agent.py` | Main agent: STT + LLM insights + document reference |
| `hybrid_retriever.py` | BM25 + embedding search with RRF fusion (~20ms) |
| `document_referencer.py` | Phase 3: Pre-filter + Retrieval + LLM alignment + Dedupe |
| `document_api.py` | FastAPI for document upload + Supabase processing |
| `document_processor.py` | PDF parsing + embeddings |
| `persistent_store.py` | SQLite/Redis document storage |
| `supabase_client.py` | Supabase Storage client for downloading PDFs |
| `prompts/document_reference.py` | LLM alignment prompt for reference validation |

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
- `MIN_SEGMENT_WORDS = 4` - Pre-filter threshold for retrieval (lowered from 6)
- `RRF_K = 60` - Reciprocal Rank Fusion constant
- `MIN_ALIGNMENT_CONFIDENCE = 0.7` - LLM alignment confidence threshold
- `ALIGNMENT_TIMEOUT_SECONDS = 4.0` - LLM timeout (increased from 2.0 for Azure latency)
- `ALIGNMENT_MAX_RETRIES = 1` - Retry attempts (2 total attempts, 8s worst-case)
- `MAX_CONCURRENT_ALIGNMENTS = 3` - Backpressure limit
- `DEDUPE_TTL_MINUTES = 5` - Reference deduplication TTL
- `STOP_PHRASE_MAX_WORDS = 12` - Only apply stop phrase filter for segments shorter than this

## STT Configuration

### Deepgram STT (WebSocket Streaming)
The agent uses `STT` from `livekit.plugins.deepgram` with its native `.stream()` method:
- **Native WebSocket streaming** via `wss://api.deepgram.com/v1/listen`
- **Automatic KeepAlive messages** every 5 seconds
- **Built-in reconnection handling**
- **Nova-3 model** with keyterms support

Current settings (optimized for meeting transcription):
```python
STT(
    model="nova-3",           # Best accuracy model
    language="en-US",
    punctuate=True,           # Better for readability
    smart_format=True,        # Format numbers, dates
    endpointing_ms=800,       # Wait 800ms silence before end of speech
    filler_words=True,        # Include "um", "uh"
    interim_results=True,     # Get partial results
)
```

**Note**: Previously used `StreamAdapter` wrapper which batches audio via HTTP POST,
causing intermittent `BrokenPipeError`. Now using `STT.stream()` directly which uses
native WebSocket streaming.
