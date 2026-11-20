# ── Stage 1: Compile Tailwind CSS with Node.js ──────────────────────────────
FROM node:22-slim AS tailwind

WORKDIR /build

# Install tailwindcss v4 via npm (includes full theme, unlike standalone CLI)
RUN npm install tailwindcss @tailwindcss/cli

# Copy only what the Tailwind compiler needs to scan for classes
COPY app.css.src .
COPY templates/ templates/
COPY static/ static/

RUN npx @tailwindcss/cli -i app.css.src -o static/app.css --minify

# ── Stage 2: Python application ─────────────────────────────────────────────
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ="Europe/Copenhagen"

# Install uv binary directly from the official image (faster than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy static assets and compiled CSS from Tailwind stage
COPY static/ static/
COPY templates/ templates/
COPY --from=tailwind /build/static/app.css static/app.css

# Copy application source
COPY app/ app/

# Create non-root user and data directories owned by it.
# Using a fixed UID so PVC mounts in k8s can match.
# Home dir is created so libraries that touch ~/.cache (httpx, aiohttp, etc) don't EPERM.
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser \
        --create-home --home-dir /home/appuser \
        --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/audio \
    && chown -R appuser:appuser /data /app /home/appuser

ENV HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache \
    XDG_CONFIG_HOME=/home/appuser/.config

# Environment defaults
ENV PROTECT_HOST="argos.local" \
    PROTECT_PORT="443" \
    WHISPER_URL="http://whisper-server:8000" \
    DATABASE_PATH="/data/transcriptions.db" \
    AUDIO_PATH="/data/audio" \
    AUDIO_BUFFER_BEFORE="5" \
    AUDIO_BUFFER_AFTER="10"

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/livez || exit 1

# Last, so a new version only changes this tiny layer.
ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION

# app package exposes the FastAPI instance as app:app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
