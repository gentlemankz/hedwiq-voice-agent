# syntax=docker/dockerfile:1.7

# =============================================================================
# STAGE 1: Base with system dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for:
# - PyMuPDF (PDF processing): build-essential, libffi-dev
# - General: curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# STAGE 2: Dependencies
# =============================================================================
FROM base AS deps

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install with no cache to reduce image size
# Using --no-compile to skip bytecode compilation (done at runtime)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# STAGE 3: Production Runner
# =============================================================================
FROM python:3.12-slim-bookworm AS runner

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only runtime system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (MANDATORY for production)
# Using --no-log-init to avoid issues with large UIDs
RUN groupadd --gid 1001 agent && \
    useradd --no-log-init --uid 1001 --gid agent --shell /bin/bash agent

# Copy installed Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code with correct ownership
COPY --chown=agent:agent . .

# Create data directory for runtime storage (SQLite, temp files)
RUN mkdir -p /app/data && chown agent:agent /app/data

# Switch to non-root user
USER agent

# Expose Document API port
EXPOSE 8000

# Health check for the Document API
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs the main LiveKit agent
# Can be overridden in docker-compose for document_api
CMD ["python", "luframe_agent.py", "start"]
