# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hedwiq Agent is a Python-based LiveKit agent that provides real-time transcription and AI-powered insight extraction for meetings. It joins LiveKit rooms as an invisible participant and:
- Transcribes all participants using Deepgram Nova-3 STT
- Extracts insights (ideas, problems, solutions, risks, etc.) using Azure OpenAI
- Publishes results via LiveKit text streams
- Supports document upload for reference detection during meetings

## Commands

### Running the Agent

```bash
# Development mode (full agent with transcription + insights)
python hedwiq_agent.py dev

# Transcription only mode
python transcription_agent.py dev

# Production mode
python hedwiq_agent.py start
```

### Running the Document API

```bash
# Start document upload API server
python document_api.py

# Or with custom port
DOCUMENT_API_PORT=8080 python document_api.py
```

### Environment Setup

```bash
# Create virtual environment (requires Python 3.12 or 3.13 - NOT 3.14)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Architecture

### Core Components

- **hedwiq_agent.py**: Main unified agent handling STT + LLM insight extraction in one process
  - `HedwiqAgent`: Orchestrates transcription and insight analysis
  - `ParticipantTranscriber`: Handles per-participant audio transcription with VAD
  - `InsightAnalyzer`: Queue-based LLM analysis with deduplication

- **document_api.py**: FastAPI server for document upload/management (runs separately from LiveKit agent)

- **document_processor.py**: PDF parsing (PyMuPDF), segmentation, and embedding generation

- **persistent_store.py**: Document storage with SQLite (dev) or Redis (prod) backends

### Data Flow

1. Audio streams from participants → `ParticipantTranscriber`
2. Silero VAD detects speech boundaries → Deepgram STT produces transcripts
3. Transcripts buffered in `InsightAnalyzer` → Azure OpenAI extracts insights
4. Results published via LiveKit text streams:
   - `lk.transcription`: Real-time transcriptions
   - `hedwiq.insight`: Extracted insights (JSON)

### Key Patterns

- **VAD-wrapped STT**: Uses `stt.StreamAdapter` with Silero VAD to prevent transcription fragmentation. Meeting-optimized with 1.2s silence duration.
- **Queue-based analysis**: Insights analyzed after speech pauses, not on every transcript
- **Deduplication**: Deterministic fingerprints + semantic similarity checking prevents duplicate insights

## Schemas

- **schemas/insights.py**: `Insight` model with types (idea, problem, solution, risk, insight, hypothesis, action_item, open_question)
- **schemas/documents.py**: `DocumentSegment`, `BoundingBox`, `UploadedDocument` for PDF reference feature

## Configuration

Required environment variables in `.env`:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`

## Key Constants

- `MIN_CONFIDENCE_THRESHOLD = 0.75`: Minimum confidence for publishing insights
- `MIN_INSIGHT_WORDS = 8`: Minimum words for valid insight content
- `MAX_DOCUMENTS_PER_ROOM = 10`: Document upload limit per room
- `DOCUMENT_TTL_HOURS = 24`: Auto-cleanup for stored documents
