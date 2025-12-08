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
```

## Key Files

| File | Purpose |
|------|---------|
| `hedwiq_agent.py` | Main agent: STT + LLM insights + document reference |
| `hybrid_retriever.py` | BM25 + embedding search with RRF fusion (~20ms) |
| `document_referencer.py` | Phase 3: Pre-filter + Retrieval + LLM alignment + Dedupe |
| `document_api.py` | FastAPI for document upload |
| `document_processor.py` | PDF parsing + embeddings |
| `persistent_store.py` | SQLite/Redis document storage |
| `prompts/document_reference.py` | LLM alignment prompt for reference validation |

## Configuration

Required in `.env`:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
- `INTERNAL_SERVICE_TOKEN` (for document API)

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
