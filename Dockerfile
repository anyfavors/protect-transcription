# Protect Transcribe Service
# Receives UniFi Protect webhooks, fetches audio, transcribes with Whisper

FROM python:3.12-slim

LABEL maintainer="homelab"
LABEL description="UniFi Protect speech transcription service"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/

# Create data directories
RUN mkdir -p /data/audio

# Environment defaults
ENV PROTECT_HOST="argos.local" \
    PROTECT_PORT="443" \
    WHISPER_URL="http://whisper-server:8000" \
    DATABASE_PATH="/data/transcriptions.db" \
    AUDIO_PATH="/data/audio" \
    AUDIO_BUFFER_BEFORE="5" \
    AUDIO_BUFFER_AFTER="10" \
    TZ="Europe/Copenhagen"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
