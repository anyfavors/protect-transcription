#!/usr/bin/env python3
"""
Protect Transcribe Service

Receives webhooks from UniFi Protect on speech detection,
fetches audio from the NVR, transcribes with Whisper,
and provides a searchable web UI.
"""

import asyncio
import hashlib
import io
import logging
import os
import sqlite3
import subprocess
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from uiprotect import ProtectApiClient

# Configuration from environment
PROTECT_HOST = os.getenv("PROTECT_HOST", "argos.local")
PROTECT_PORT = int(os.getenv("PROTECT_PORT", "443"))
PROTECT_USERNAME = os.getenv("PROTECT_USERNAME", "")
PROTECT_PASSWORD = os.getenv("PROTECT_PASSWORD", "")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper-server:8000")
DATABASE_PATH = os.getenv("DATABASE_PATH", "/data/transcriptions.db")
AUDIO_PATH = os.getenv("AUDIO_PATH", "/data/audio")
# How many seconds before/after the event timestamp to capture
AUDIO_BUFFER_BEFORE = int(os.getenv("AUDIO_BUFFER_BEFORE", "5"))
AUDIO_BUFFER_AFTER = int(os.getenv("AUDIO_BUFFER_AFTER", "10"))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Protect API client
protect_client: Optional[ProtectApiClient] = None


def init_database():
    """Initialize SQLite database with required tables."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            camera_id TEXT,
            camera_name TEXT,
            timestamp DATETIME,
            transcription TEXT,
            language TEXT,
            confidence REAL,
            audio_file TEXT,
            duration_seconds REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_camera ON transcriptions(camera_name)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_status ON transcriptions(status)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DATABASE_PATH}")


async def get_protect_client() -> ProtectApiClient:
    """Get or create the Protect API client."""
    global protect_client
    
    if protect_client is None:
        logger.info(f"Connecting to UniFi Protect at {PROTECT_HOST}")
        protect_client = ProtectApiClient(
            host=PROTECT_HOST,
            port=PROTECT_PORT,
            username=PROTECT_USERNAME,
            password=PROTECT_PASSWORD,
            verify_ssl=False  # Local network, self-signed cert
        )
        await protect_client.update()
        logger.info("Connected to UniFi Protect")
    
    return protect_client


async def fetch_audio_clip(
    camera_id: str,
    start_time: datetime,
    end_time: datetime
) -> Optional[bytes]:
    """
    Fetch audio clip from UniFi Protect NVR.
    
    The Protect API provides video+audio, so we'll need to
    extract audio using ffmpeg.
    """
    try:
        client = await get_protect_client()
        
        # Get camera by ID
        camera = client.bootstrap.cameras.get(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return None
        
        logger.info(f"Fetching clip from {camera.name} ({start_time} to {end_time})")
        
        # Use the API's get_camera_video method to download video
        # This returns the video as bytes
        video_data = await client.get_camera_video(
            camera_id,
            start_time,
            end_time,
            channel=0  # Main channel (highest quality with audio)
        )
        
        if not video_data:
            logger.error("No video data received from Protect")
            return None
        
        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_file.write(video_data)
            video_path = video_file.name
        
        audio_path = video_path.replace(".mp4", ".wav")
        
        try:
            # Extract audio to WAV (16kHz mono - optimal for Whisper)
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # 16kHz
                "-ac", "1",  # Mono
                audio_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return None
            
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            return audio_data
            
        finally:
            # Cleanup temp files
            Path(video_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.exception(f"Error fetching audio clip: {e}")
        return None


async def transcribe_audio(audio_data: bytes, language: str = "da") -> dict:
    """
    Send audio to Whisper server for transcription.
    Uses the OpenAI-compatible API.
    """
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare multipart form data
            files = {
                "file": ("audio.wav", audio_data, "audio/wav")
            }
            data = {
                "model": "large-v3",  # Will be ignored, server uses configured model
                "language": language,
                "response_format": "verbose_json"
            }
            
            response = await client.post(
                f"{WHISPER_URL}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                logger.error(f"Whisper API error: {response.status_code} - {response.text}")
                return {"error": response.text}
            
            result = response.json()
            logger.info(f"Transcription result: {result.get('text', '')[:100]}...")
            return result
            
    except Exception as e:
        logger.exception(f"Error calling Whisper API: {e}")
        return {"error": str(e)}


async def process_speech_event(
    event_id: str,
    camera_id: str,
    timestamp_ms: int
):
    """
    Process a speech detection event:
    1. Fetch audio from NVR
    2. Transcribe with Whisper
    3. Store in database
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if already processed
        cursor.execute("SELECT id FROM transcriptions WHERE event_id = ?", (event_id,))
        if cursor.fetchone():
            logger.info(f"Event {event_id} already processed, skipping")
            return
        
        # Get camera info
        client = await get_protect_client()
        camera = client.bootstrap.cameras.get(camera_id)
        camera_name = camera.name if camera else "Unknown"
        
        # Calculate time range
        event_time = datetime.fromtimestamp(timestamp_ms / 1000)
        start_time = event_time - timedelta(seconds=AUDIO_BUFFER_BEFORE)
        end_time = event_time + timedelta(seconds=AUDIO_BUFFER_AFTER)
        
        # Insert pending record
        cursor.execute("""
            INSERT INTO transcriptions (event_id, camera_id, camera_name, timestamp, status)
            VALUES (?, ?, ?, ?, 'processing')
        """, (event_id, camera_id, camera_name, event_time.isoformat()))
        conn.commit()
        
        # Fetch audio
        audio_data = await fetch_audio_clip(camera_id, start_time, end_time)
        
        if not audio_data:
            cursor.execute("""
                UPDATE transcriptions SET status = 'error', transcription = 'Failed to fetch audio'
                WHERE event_id = ?
            """, (event_id,))
            conn.commit()
            return
        
        # Save audio file
        Path(AUDIO_PATH).mkdir(parents=True, exist_ok=True)
        audio_hash = hashlib.md5(audio_data).hexdigest()[:8]
        audio_filename = f"{event_time.strftime('%Y%m%d_%H%M%S')}_{camera_name}_{audio_hash}.wav"
        audio_filepath = Path(AUDIO_PATH) / audio_filename
        
        with open(audio_filepath, "wb") as f:
            f.write(audio_data)
        
        # Transcribe (Danish by default)
        result = await transcribe_audio(audio_data, language="da")
        
        if "error" in result:
            cursor.execute("""
                UPDATE transcriptions 
                SET status = 'error', transcription = ?, audio_file = ?
                WHERE event_id = ?
            """, (f"Transcription error: {result['error']}", audio_filename, event_id))
        else:
            # Calculate duration from audio file
            duration = len(audio_data) / (16000 * 2)  # 16kHz, 16-bit
            
            cursor.execute("""
                UPDATE transcriptions 
                SET status = 'completed',
                    transcription = ?,
                    language = ?,
                    confidence = ?,
                    audio_file = ?,
                    duration_seconds = ?
                WHERE event_id = ?
            """, (
                result.get("text", ""),
                result.get("language", "da"),
                result.get("confidence", 0),
                audio_filename,
                duration,
                event_id
            ))
        
        conn.commit()
        logger.info(f"Processed event {event_id} from {camera_name}")
        
    except Exception as e:
        logger.exception(f"Error processing event {event_id}: {e}")
        cursor.execute("""
            UPDATE transcriptions SET status = 'error', transcription = ?
            WHERE event_id = ?
        """, (str(e), event_id))
        conn.commit()
    finally:
        conn.close()


# Pydantic models
class WebhookPayload(BaseModel):
    """UniFi Protect webhook payload structure."""
    alarm: dict
    timestamp: int


class TranscriptionResponse(BaseModel):
    id: int
    event_id: str
    camera_name: str
    timestamp: str
    transcription: str
    language: Optional[str]
    duration_seconds: Optional[float]
    status: str
    audio_file: Optional[str]


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_database()
    Path(AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    
    # Try to connect to Protect on startup
    try:
        await get_protect_client()
    except Exception as e:
        logger.warning(f"Could not connect to Protect on startup: {e}")
    
    yield
    
    # Shutdown
    global protect_client
    if protect_client:
        await protect_client.close()


app = FastAPI(
    title="Protect Transcribe",
    description="Speech transcription service for UniFi Protect",
    lifespan=lifespan
)

# Templates
templates = Jinja2Templates(directory="/app/templates")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Receive webhook from UniFi Protect Alarm Manager.
    
    Expected payload format:
    {
        "alarm": {
            "name": "Speech Detection",
            "sources": [{"device": "CAMERA_MAC", "type": "include"}],
            "triggers": [{"key": "speech", "device": "CAMERA_MAC"}]
        },
        "timestamp": 1725883107267
    }
    """
    try:
        payload = await request.json()
        logger.info(f"Received webhook: {payload}")
        
        alarm = payload.get("alarm", {})
        timestamp = payload.get("timestamp", 0)
        
        # Extract camera ID and trigger type from payload
        triggers = alarm.get("triggers", [])
        
        for trigger in triggers:
            trigger_key = trigger.get("key", "")
            camera_id = trigger.get("device", "")
            
            # Only process speech events
            if trigger_key.lower() in ["speech", "voice", "talking"]:
                # Generate unique event ID
                event_id = f"{camera_id}_{timestamp}_{trigger_key}"
                
                # Process in background
                background_tasks.add_task(
                    process_speech_event,
                    event_id,
                    camera_id,
                    timestamp
                )
                
                logger.info(f"Queued speech event {event_id} for processing")
        
        return {"status": "accepted", "message": "Webhook received"}
        
    except Exception as e:
        logger.exception(f"Error processing webhook: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/transcriptions")
async def get_transcriptions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    camera: Optional[str] = None,
    date: Optional[str] = None,
    search: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Get transcriptions with filtering and pagination.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Build query
        where_clauses = []
        params = []
        
        if camera:
            where_clauses.append("camera_name = ?")
            params.append(camera)
        
        if date:
            where_clauses.append("DATE(timestamp) = ?")
            params.append(date)
        
        if search:
            where_clauses.append("transcription LIKE ?")
            params.append(f"%{search}%")
        
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM transcriptions WHERE {where_sql}", params)
        total = cursor.fetchone()[0]
        
        # Get paginated results
        offset = (page - 1) * per_page
        cursor.execute(f"""
            SELECT * FROM transcriptions 
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset])
        
        rows = cursor.fetchall()
        
        transcriptions = [
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "camera_name": row["camera_name"],
                "timestamp": row["timestamp"],
                "transcription": row["transcription"],
                "language": row["language"],
                "duration_seconds": row["duration_seconds"],
                "status": row["status"],
                "audio_file": row["audio_file"]
            }
            for row in rows
        ]
        
        return {
            "transcriptions": transcriptions,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
        
    finally:
        conn.close()


@app.get("/api/cameras")
async def get_cameras():
    """Get list of cameras that have transcriptions."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT DISTINCT camera_name 
            FROM transcriptions 
            WHERE camera_name IS NOT NULL
            ORDER BY camera_name
        """)
        
        cameras = [row[0] for row in cursor.fetchall()]
        return {"cameras": cameras}
        
    finally:
        conn.close()


@app.get("/api/dates")
async def get_dates():
    """Get list of dates that have transcriptions."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT DISTINCT DATE(timestamp) as date
            FROM transcriptions 
            WHERE timestamp IS NOT NULL
            ORDER BY date DESC
            LIMIT 90
        """)
        
        dates = [row[0] for row in cursor.fetchall()]
        return {"dates": dates}
        
    finally:
        conn.close()


@app.get("/api/stats")
async def get_stats():
    """Get transcription statistics."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM transcriptions")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'completed'")
        completed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'processing'")
        processing = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'error'")
        errors = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM transcriptions 
            WHERE DATE(timestamp) = DATE('now')
        """)
        today = cursor.fetchone()[0]
        
        return {
            "total": total,
            "completed": completed,
            "processing": processing,
            "errors": errors,
            "today": today
        }
        
    finally:
        conn.close()


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    file_path = Path(AUDIO_PATH) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# Manual test endpoint (for development)
@app.post("/api/test")
async def test_transcription(background_tasks: BackgroundTasks):
    """
    Test endpoint to manually trigger a transcription.
    Useful for testing without actual webhook.
    """
    try:
        client = await get_protect_client()
        
        # Get first camera
        cameras = list(client.bootstrap.cameras.values())
        if not cameras:
            return {"error": "No cameras found"}
        
        camera = cameras[0]
        event_id = f"test_{int(datetime.now().timestamp() * 1000)}"
        timestamp = int(datetime.now().timestamp() * 1000)
        
        background_tasks.add_task(
            process_speech_event,
            event_id,
            camera.id,
            timestamp
        )
        
        return {
            "status": "queued",
            "event_id": event_id,
            "camera": camera.name
        }
        
    except Exception as e:
        logger.exception(f"Test error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
