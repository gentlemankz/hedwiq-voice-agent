# Hedwiq Agent

A Python-based LiveKit agent that provides real-time transcription and AI-powered insight extraction for Hedwiq meetings.

## Overview

This agent joins LiveKit rooms as an invisible participant and:
- Listens to all participant audio tracks simultaneously
- Performs real-time speech-to-text transcription using Deepgram Nova-2
- Analyzes transcripts with Azure OpenAI to extract insights
- Publishes transcriptions via LiveKit text streams (`lk.transcription` topic)
- Publishes insights via LiveKit text streams (`hedwiq.insight` topic)
- The frontend receives both streams and displays them in the sidebar

## Features

### Real-time Transcription
- Multi-participant support: Transcribes all participants simultaneously
- Speaker identification: Each transcription includes the speaker's identity and name
- Real-time streaming: Transcriptions appear as speech happens
- Interim results: Shows partial transcriptions while speaking

### AI-Powered Insights (Phase 2)
The agent uses Azure OpenAI to detect and extract insights in real-time:

| Type | Description | Example Triggers |
|------|-------------|-----------------|
| **Idea** | New suggestions or proposals | "We could...", "What if we..." |
| **Problem** | Issues, challenges, pain points | "The problem is...", "We're struggling with..." |
| **Solution** | Proposed fixes | "Let's fix this by...", "The solution is..." |
| **Risk** | Concerns, limitations | "This might...", "I'm worried about..." |
| **Insight** | Key observations | "I noticed...", "The data shows..." |
| **Hypothesis** | Assumptions to validate | "I think...", "My guess is..." |
| **Action Item** | Tasks requiring follow-up | "John will...", "By Friday we need to..." |
| **Open Question** | Unresolved questions | "How will we...?", "What about...?" |

## Setup

### Prerequisites

- **Python 3.12 or 3.13** (required - Python 3.14 is NOT supported by livekit-agents)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- A Deepgram API key (https://console.deepgram.com/)
- Azure OpenAI credentials (https://portal.azure.com)

> **Important**: The LiveKit Agents SDK does not support Python 3.14. If you have Python 3.14 installed, you need to install Python 3.12 or 3.13 to run this agent.

### Installation with uv (recommended)

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment and install dependencies:
   ```bash
   cd agent
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

### Installation with pip

```bash
cd agent
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Update the `.env` file with your credentials:

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Deepgram API Key for Speech-to-Text
DEEPGRAM_API_KEY=your_deepgram_api_key_here
# Optional: STT tuning (defaults shown)
STT_MODEL=nova-3
STT_LANGUAGE=en  # set to 'multi' for multilingual meetings

# Azure OpenAI Configuration for Insight Extraction
# These env var names are auto-detected by the LiveKit OpenAI plugin
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

## Running

### Development Mode (Full Agent with Insights)

```bash
python hedwiq_agent.py dev
```

This runs the full Hedwiq agent with both transcription and insight extraction.

### Transcription Only Mode

If you only need transcription without AI insights:

```bash
python transcription_agent.py dev
```

### Production Mode

```bash
python hedwiq_agent.py start
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LiveKit Room                            │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ Participant │ Participant │ Participant │   Hedwiq Agent   │
│   (User A)  │   (User B)  │   (User C)  │   (Invisible)    │
└──────┬──────┴──────┬──────┴──────┬──────┴────────┬─────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                    Audio Streams (subscribed by agent)
                           │
                           ▼
       ┌───────────────────────────────────────────┐
       │            Hedwiq Agent Pipeline          │
       ├───────────────────────────────────────────┤
       │  ┌─────────────────────────────────────┐  │
       │  │   Multi-Participant Transcriber     │  │
       │  │   AudioStream → Deepgram STT        │  │
       │  └──────────────────┬──────────────────┘  │
       │                     │                     │
       │                     ▼                     │
       │        lk.transcription text stream      │
       │                     │                     │
       │                     ▼                     │
       │  ┌─────────────────────────────────────┐  │
       │  │       Insight Analyzer              │  │
       │  │    Transcript Buffer → Azure LLM   │  │
       │  └──────────────────┬──────────────────┘  │
       │                     │                     │
       │                     ▼                     │
       │        hedwiq.insight text stream        │
       └───────────────────────────────────────────┘
                           │
                           ▼
                    Frontend Sidebar UI
                  (Transcript + Insights)
```

## Text Stream Topics

| Topic | Purpose | Data Format |
|-------|---------|-------------|
| `lk.transcription` | Real-time transcription | Plain text with attributes |
| `hedwiq.insight` | Detected insights | JSON: `{type, content, speaker, confidence}` |

## Troubleshooting

### Agent not receiving audio
- Ensure participants have granted microphone permissions in their browser
- Verify audio tracks are being published (check browser console for errors)
- Confirm the agent is connected (check agent logs)

### Transcriptions not appearing
- Verify the agent is connected and running (check terminal output)
- Ensure DEEPGRAM_API_KEY is valid in `.env`
- Check that the frontend is subscribed to `lk.transcription` topic
- Verify LiveKit API credentials are correct

### Insights not appearing
- Verify Azure OpenAI credentials are correct
- Check that AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set
- Ensure the deployment name (AZURE_OPENAI_DEPLOYMENT) matches your Azure setup
- Check agent logs for LLM errors

### High latency
- Consider using a LiveKit region closer to your users
- Ensure stable network connection
- Deepgram Nova-2 is optimized for low latency
- Azure OpenAI gpt-4o-mini provides fast inference for insights

### No agent logs appearing
- Make sure you're running in `dev` mode, not `start`
- Check that LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are correct

## Document Reference Feature (Phase 1)

The agent now supports real-time document reference detection. Admins can upload PDF documents, and the system automatically detects when speakers reference content from those documents.

### Running the Document API

In addition to the main agent, you can run a separate HTTP API server for document upload:

```bash
# Start the document upload API (default port 8000)
python document_api.py

# Or with custom port
DOCUMENT_API_PORT=8080 python document_api.py
```

The document API provides:
- `POST /documents/upload` - Upload and process PDF documents
- `GET /documents/{id}/pdf` - Retrieve PDF file content
- `GET /documents/{id}` - Get document metadata
- `GET /documents/room/{room_id}` - List documents for a room
- `DELETE /documents/{id}` - Delete a document
- `GET /health` - Health check endpoint

### Document Processing Pipeline

1. **PDF Parsing**: Extract text with bounding box coordinates using PyMuPDF
2. **Segmentation**: Split into retrieval-optimized segments (500 chars max)
3. **Embedding**: Generate semantic embeddings using Azure OpenAI `text-embedding-3-large`
4. **Storage**: Persist to SQLite (development) or Redis (production)

### Configuration for Documents

Add to your `.env`:

```bash
# Document API Authentication (for frontend → agent communication)
INTERNAL_SERVICE_TOKEN=your_secure_token_here

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000

# Document API Port (optional, default 8000)
DOCUMENT_API_PORT=8000
```

## File Structure

```
agent/
├── hedwiq_agent.py          # Main agent with transcription + insights
├── transcription_agent.py   # Transcription-only agent
├── document_api.py          # HTTP API for document upload (NEW)
├── document_processor.py    # PDF parsing + embeddings (NEW)
├── persistent_store.py      # Document storage (SQLite/Redis) (NEW)
├── schemas/
│   ├── __init__.py
│   ├── insights.py          # Insight data models
│   └── documents.py         # Document data models (NEW)
├── prompts/
│   ├── __init__.py
│   └── insight_extraction.py # LLM prompts for insight extraction
├── requirements.txt
├── .env                     # Environment variables (not in git)
└── README.md
```

## Performance Considerations

### Latency Budget

| Component | Target Latency |
|-----------|----------------|
| STT (Deepgram) | < 300ms |
| LLM Analysis (Azure) | < 2000ms |
| Text Stream Delivery | < 100ms |
| Frontend Render | < 50ms |
| **Total Insight Latency** | **< 2500ms** |

### Cost Optimization

- Uses `gpt-4o-mini` for fast, cost-effective inference
- Only analyzes final transcripts, not interim
- Debounces analysis to avoid excessive LLM calls
- Sets confidence threshold to filter low-quality insights
