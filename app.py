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
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

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
# Timezone for datetime conversion (should match your Protect NVR timezone)
LOCAL_TZ = ZoneInfo(os.getenv("TZ", "Europe/Copenhagen"))

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Protect API client
protect_client: Optional[ProtectApiClient] = None


def init_database():
    """Initialize SQLite database with required tables and FTS5 for full-text search."""
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
            segments TEXT,
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
    
    # FTS5 virtual table for fast full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
            transcription,
            camera_name,
            content='transcriptions',
            content_rowid='id'
        )
    """)
    
    # Triggers to keep FTS index in sync with main table
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ai AFTER INSERT ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(rowid, transcription, camera_name)
            VALUES (new.id, new.transcription, new.camera_name);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ad AFTER DELETE ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, camera_name)
            VALUES ('delete', old.id, old.transcription, old.camera_name);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_au AFTER UPDATE ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, camera_name)
            VALUES ('delete', old.id, old.transcription, old.camera_name);
            INSERT INTO transcriptions_fts(rowid, transcription, camera_name)
            VALUES (new.id, new.transcription, new.camera_name);
        END
    """)
    
    # Settings table for configurable parameters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Initialize default settings if not present
    # Use env vars as defaults for backwards compatibility
    default_settings = {
        'whisper_model': 'Systran/faster-whisper-large-v3',
        'language': 'da',
        'buffer_before': '5',
        'buffer_after': '60',  # 1 minute default, adjustable up to 10 min
        'vad_filter': 'true',
        'beam_size': '5',
        'protect_host': os.getenv("PROTECT_HOST", ""),  # NVR address
    }
    
    for key, value in default_settings.items():
        cursor.execute("""
            INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)
        """, (key, value))
    
    # Migration: Add segments column if it doesn't exist
    cursor.execute("PRAGMA table_info(transcriptions)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'segments' not in columns:
        logger.info("Migrating database: adding segments column")
        cursor.execute("ALTER TABLE transcriptions ADD COLUMN segments TEXT")
    
    # Rebuild FTS index to include existing data (idempotent)
    try:
        cursor.execute("INSERT INTO transcriptions_fts(transcriptions_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        pass  # Table might be empty or already synced
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DATABASE_PATH} with FTS5 support")


def get_settings() -> dict:
    """Get all settings from database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    conn.close()
    
    settings = {row[0]: row[1] for row in rows}
    return settings


def get_setting(key: str, default: str = None) -> str:
    """Get a single setting from database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    
    return row[0] if row else default


def save_setting(key: str, value: str):
    """Save a setting to database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO settings (key, value, updated_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (key, value))
    
    conn.commit()
    conn.close()
    logger.info(f"Setting saved: {key} = {value}")


def get_protect_host() -> str:
    """Get Protect host from settings, fall back to env var."""
    host = get_setting('protect_host', '')
    if not host:
        host = PROTECT_HOST
    return host


async def get_protect_client(force_reconnect: bool = False) -> ProtectApiClient:
    """Get or create the Protect API client with reconnection support."""
    global protect_client
    
    host = get_protect_host()
    if not host:
        raise ValueError("Protect host not configured. Set it in Settings.")
    
    if protect_client is None or force_reconnect:
        if protect_client is not None:
            try:
                await protect_client.close()
            except Exception:
                pass
        
        logger.info(f"Connecting to UniFi Protect at {host}")
        protect_client = ProtectApiClient(
            host=host,
            port=PROTECT_PORT,
            username=PROTECT_USERNAME,
            password=PROTECT_PASSWORD,
            verify_ssl=False  # Local network, self-signed cert
        )
        await protect_client.update()
        logger.info("Connected to UniFi Protect")
    else:
        # Check if we need to refresh the connection
        try:
            # Try to access bootstrap to verify connection is alive
            _ = protect_client.bootstrap.nvr.name
        except Exception as e:
            logger.warning(f"Protect client connection stale, reconnecting: {e}")
            try:
                await protect_client.close()
            except Exception:
                pass
            protect_client = ProtectApiClient(
                host=host,
                port=PROTECT_PORT,
                username=PROTECT_USERNAME,
                password=PROTECT_PASSWORD,
                verify_ssl=False
            )
            await protect_client.update()
            logger.info("Reconnected to UniFi Protect")
    
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
    
    camera_id can be either a UUID or MAC address.
    """
    try:
        client = await get_protect_client()
        
        # Find camera - could be by UUID or MAC address
        camera = None
        
        # First try direct lookup (UUID)
        camera = client.bootstrap.cameras.get(camera_id)
        
        # If not found, try MAC address lookup
        if not camera:
            # Normalize MAC address (remove colons/dashes, uppercase)
            normalized_mac = camera_id.upper().replace(":", "").replace("-", "")
            for cam in client.bootstrap.cameras.values():
                cam_mac = cam.mac.upper().replace(":", "").replace("-", "")
                if cam_mac == normalized_mac:
                    camera = cam
                    logger.info(f"Found camera by MAC: {cam.name} ({cam.id})")
                    break
        
        if not camera:
            logger.error(f"Camera {camera_id} not found (tried UUID and MAC lookup)")
            # Log available cameras for debugging
            logger.info(f"Available cameras: {[(c.name, c.mac, c.id) for c in client.bootstrap.cameras.values()]}")
            return None
        
        logger.info(f"Fetching clip from {camera.name} ({start_time.isoformat()} to {end_time.isoformat()})")
        
        # Try to export video using the Camera object
        # Method signature varies by uiprotect version
        video_data = None
        
        try:
            # Try the most common method names
            if hasattr(camera, 'get_video'):
                logger.debug("Using camera.get_video()")
                video_data = await camera.get_video(start_time, end_time)
            elif hasattr(camera, 'export_video'):
                logger.debug("Using camera.export_video()")
                video_data = await camera.export_video(start_time, end_time)
            else:
                # List available methods for debugging
                video_methods = [m for m in dir(camera) if 'video' in m.lower() or 'export' in m.lower()]
                logger.error(f"No video export method found. Available video-related methods: {video_methods}")
                logger.debug(f"All camera methods: {[m for m in dir(camera) if not m.startswith('_')]}")
                return None
                
        except TypeError as e:
            # Method might have different signature - try with output_file
            logger.warning(f"Method call failed with TypeError: {e}, trying alternative signatures")
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                if hasattr(camera, 'get_video'):
                    await camera.get_video(start_time, end_time, output_file=tmp_path)
                    video_data = tmp_path.read_bytes()
                    tmp_path.unlink(missing_ok=True)
            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                raise
        except Exception as video_err:
            logger.error(f"Error calling video export method: {video_err}")
            raise
        
        if not video_data:
            logger.error("No video data received from Protect")
            return None
        
        logger.info(f"Received {len(video_data)} bytes of video data")
        
        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_file.write(video_data)
            video_path = video_file.name
        
        audio_path = video_path.replace(".mp4", ".wav")
        
        try:
            # Extract audio to WAV (16kHz mono - optimal for Whisper)
            # Audio filters: highpass removes low-frequency rumble, loudnorm normalizes volume
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # 16kHz
                "-ac", "1",  # Mono
                "-af", "highpass=f=200,loudnorm",  # Remove rumble, normalize volume
                audio_path
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error (exit code {result.returncode}): {result.stderr}")
                # Check if it's a "no audio stream" error
                if "does not contain any stream" in result.stderr or "Output file is empty" in result.stderr:
                    logger.error("Video file contains no audio stream")
                return None
            
            # Check if audio file was created and has content
            audio_file_path = Path(audio_path)
            if not audio_file_path.exists() or audio_file_path.stat().st_size == 0:
                logger.error("ffmpeg produced empty or no audio file")
                return None
            
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            logger.info(f"Extracted {len(audio_data)} bytes of audio")
            return audio_data
            
        finally:
            # Cleanup temp files
            Path(video_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.exception(f"Error fetching audio clip: {e}")
        return None


async def transcribe_audio(audio_data: bytes) -> dict:
    """
    Send audio to Whisper server for transcription.
    Uses the OpenAI-compatible API with settings from database.
    """
    try:
        # Get settings from database
        settings = get_settings()
        model = settings.get('whisper_model', 'Systran/faster-whisper-large-v3')
        language = settings.get('language', 'da')
        vad_filter = settings.get('vad_filter', 'true').lower() == 'true'
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare multipart form data
            files = {
                "file": ("audio.wav", audio_data, "audio/wav")
            }
            data = {
                "model": model,
                "language": language,
                "response_format": "verbose_json",
                "temperature": "0.0",  # Deterministic output for better accuracy
                "initial_prompt": "Dette er en optagelse fra et overvågningskamera i et privat hjem. Samtalen er på dansk."
            }
            
            # Add VAD filter if enabled (helps remove silence/noise)
            if vad_filter:
                data["vad_filter"] = "true"
            
            logger.info(f"Transcribing with model={model}, language={language}, vad={vad_filter}")
            
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


def queue_transcription(
    event_id: str,
    camera_id: str,
    camera_name: str,
    timestamp_ms: int,
    language: str = 'da'
) -> bool:
    """
    Queue a transcription by inserting a pending record into the database.
    Returns True if queued, False if already exists.
    """
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    try:
        # Check if already exists
        cursor.execute("SELECT id FROM transcriptions WHERE event_id = ?", (event_id,))
        if cursor.fetchone():
            logger.debug(f"Event {event_id} already exists, skipping queue")
            return False
        
        # Calculate event time
        event_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=LOCAL_TZ)
        
        # Insert pending record - worker will pick it up
        cursor.execute("""
            INSERT INTO transcriptions (event_id, camera_id, camera_name, timestamp, status, language)
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (event_id, camera_id, camera_name, event_time.isoformat(), language))
        conn.commit()
        
        logger.info(f"Queued transcription for event {event_id} from {camera_name}")
        return True
        
    except sqlite3.IntegrityError:
        # Event already exists (race condition)
        logger.debug(f"Event {event_id} already exists (integrity error)")
        return False
    finally:
        conn.close()


async def process_pending_transcription(row: dict) -> None:
    """
    Process a single pending transcription record.
    Fetches audio, transcribes, and updates the database.
    """
    event_id = row['event_id']
    camera_id = row['camera_id']
    camera_name = row['camera_name']
    timestamp_str = row['timestamp']
    record_id = row['id']
    
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    try:
        # Mark as processing
        cursor.execute("""
            UPDATE transcriptions SET status = 'processing' WHERE id = ?
        """, (record_id,))
        conn.commit()
        
        # Get settings from database
        settings = get_settings()
        buffer_before = int(settings.get('buffer_before', '5'))
        buffer_after = int(settings.get('buffer_after', '60'))
        
        # Parse timestamp
        try:
            event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=LOCAL_TZ)
        except Exception as e:
            logger.error(f"Failed to parse timestamp {timestamp_str}: {e}")
            raise ValueError(f"Invalid timestamp: {timestamp_str}")
        
        start_time = event_time - timedelta(seconds=buffer_before)
        end_time = event_time + timedelta(seconds=buffer_after)
        
        logger.info(f"Processing event {event_id} from {camera_name} at {event_time.isoformat()}")
        logger.info(f"Using buffer: {buffer_before}s before, {buffer_after}s after")
        
        # Fetch audio
        audio_data = await fetch_audio_clip(camera_id, start_time, end_time)
        
        if not audio_data:
            cursor.execute("""
                UPDATE transcriptions SET status = 'error', transcription = 'Failed to fetch audio'
                WHERE id = ?
            """, (record_id,))
            conn.commit()
            return
        
        # Save audio file
        Path(AUDIO_PATH).mkdir(parents=True, exist_ok=True)
        audio_hash = hashlib.md5(audio_data).hexdigest()[:8]
        audio_filename = f"{event_time.strftime('%Y%m%d_%H%M%S')}_{camera_name}_{audio_hash}.wav"
        audio_filepath = Path(AUDIO_PATH) / audio_filename
        
        with open(audio_filepath, "wb") as f:
            f.write(audio_data)
        
        # Transcribe
        result = await transcribe_audio(audio_data)
        
        if "error" in result:
            cursor.execute("""
                UPDATE transcriptions 
                SET status = 'error', transcription = ?, audio_file = ?
                WHERE id = ?
            """, (f"Transcription error: {result['error']}", audio_filename, record_id))
        else:
            # Calculate duration from audio file
            duration = len(audio_data) / (16000 * 2)  # 16kHz, 16-bit
            
            # Extract and store segments with timestamps
            segments = result.get("segments", [])
            segments_json = json.dumps(segments) if segments else None
            
            cursor.execute("""
                UPDATE transcriptions 
                SET status = 'completed',
                    transcription = ?,
                    segments = ?,
                    language = ?,
                    confidence = ?,
                    audio_file = ?,
                    duration_seconds = ?
                WHERE id = ?
            """, (
                result.get("text", ""),
                segments_json,
                result.get("language", "da"),
                result.get("confidence", 0),
                audio_filename,
                duration,
                record_id
            ))
        
        conn.commit()
        logger.info(f"Completed processing event {event_id} from {camera_name}")
        
    except Exception as e:
        logger.exception(f"Error processing event {event_id}: {e}")
        try:
            cursor.execute("""
                UPDATE transcriptions SET status = 'error', transcription = ?
                WHERE id = ?
            """, (str(e)[:500], record_id))
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()


async def transcription_worker():
    """
    Background worker that polls for pending transcriptions and processes them.
    Runs in an infinite loop, processing one event at a time to avoid DB locks.
    """
    logger.info("Transcription worker started")
    
    while True:
        try:
            # Fetch one pending record
            conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, event_id, camera_id, camera_name, timestamp, language
                FROM transcriptions
                WHERE status = 'pending'
                ORDER BY timestamp ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Process this transcription
                await process_pending_transcription(dict(row))
                # Small delay between processing to avoid overwhelming the system
                await asyncio.sleep(1)
            else:
                # No pending work, wait before polling again
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.exception(f"Error in transcription worker: {e}")
            await asyncio.sleep(10)  # Wait longer on error


# Keep legacy function for compatibility with retry endpoint
async def process_speech_event(
    event_id: str,
    camera_id: str,
    timestamp_ms: int,
    skip_wait: bool = False
):
    """
    Legacy function for processing speech events.
    Now just queues the event for the worker to process.
    """
    # Get camera name
    try:
        client = await get_protect_client()
        camera = client.bootstrap.cameras.get(camera_id)
        if not camera:
            normalized_mac = camera_id.upper().replace(":", "").replace("-", "")
            for cam in client.bootstrap.cameras.values():
                cam_mac = cam.mac.upper().replace(":", "").replace("-", "")
                if cam_mac == normalized_mac:
                    camera = cam
                    break
        camera_name = camera.name if camera else f"Unknown ({camera_id})"
    except Exception:
        camera_name = f"Unknown ({camera_id})"
    
    settings = get_settings()
    language = settings.get('language', 'da')
    
    queue_transcription(event_id, camera_id, camera_name, timestamp_ms, language)


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
    
    # Start the transcription worker as a background task
    worker_task = asyncio.create_task(transcription_worker())
    logger.info("Started transcription worker task")
    
    yield
    
    # Shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Transcription worker cancelled")
    
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
async def receive_webhook(request: Request):
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
        
        # Get settings for language
        settings = get_settings()
        language = settings.get('language', 'da')
        
        events_queued = 0
        
        for trigger in triggers:
            trigger_key = trigger.get("key", "")
            camera_id = trigger.get("device", "")
            event_id_from_protect = trigger.get("eventId", "")
            trigger_timestamp = trigger.get("timestamp", timestamp)
            
            logger.debug(f"Processing trigger: key={trigger_key}, device={camera_id}")
            
            # Only process speech events (audio_alarm_speak / alrmspeak is the UniFi Protect key)
            if trigger_key.lower() in ["speech", "voice", "talking", "audio_alarm_speak", "alrmspeak"]:
                # Use Protect's event ID if available, otherwise generate one
                event_id = event_id_from_protect or f"{camera_id}_{trigger_timestamp}_{trigger_key}"
                
                logger.info(f"Speech event detected: key={trigger_key}, camera={camera_id}, event_id={event_id}")
                
                # Get camera name
                try:
                    client = await get_protect_client()
                    camera = client.bootstrap.cameras.get(camera_id)
                    if not camera:
                        normalized_mac = camera_id.upper().replace(":", "").replace("-", "")
                        for cam in client.bootstrap.cameras.values():
                            cam_mac = cam.mac.upper().replace(":", "").replace("-", "")
                            if cam_mac == normalized_mac:
                                camera = cam
                                break
                    camera_name = camera.name if camera else f"Unknown ({camera_id})"
                except Exception:
                    camera_name = f"Unknown ({camera_id})"
                
                # Queue for processing (worker will pick it up)
                if queue_transcription(event_id, camera_id, camera_name, trigger_timestamp, language):
                    events_queued += 1
                    logger.info(f"Queued speech event {event_id} for processing")
            else:
                logger.debug(f"Ignoring non-speech trigger: {trigger_key}")
        
        return {"status": "accepted", "message": f"Webhook received, {events_queued} events queued"}
        
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
    Uses FTS5 for fast full-text search.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Build query - use FTS5 for search if provided
        if search:
            # Use FTS5 MATCH for fast full-text search
            # Escape special FTS5 characters
            search_term = search.replace('"', '""')
            
            # Build additional filters
            where_clauses = []
            params = []
            
            if camera:
                where_clauses.append("t.camera_name = ?")
                params.append(camera)
            
            if date:
                where_clauses.append("DATE(t.timestamp) = ?")
                params.append(date)
            
            if status:
                where_clauses.append("t.status = ?")
                params.append(status)
            
            additional_where = " AND " + " AND ".join(where_clauses) if where_clauses else ""
            
            # Get total count with FTS5
            cursor.execute(f"""
                SELECT COUNT(*) FROM transcriptions t
                INNER JOIN transcriptions_fts fts ON t.id = fts.rowid
                WHERE transcriptions_fts MATCH ?
                {additional_where}
            """, [f'"{search_term}"'] + params)
            total = cursor.fetchone()[0]
            
            # Get paginated results with FTS5
            offset = (page - 1) * per_page
            cursor.execute(f"""
                SELECT t.* FROM transcriptions t
                INNER JOIN transcriptions_fts fts ON t.id = fts.rowid
                WHERE transcriptions_fts MATCH ?
                {additional_where}
                ORDER BY t.timestamp DESC
                LIMIT ? OFFSET ?
            """, [f'"{search_term}"'] + params + [per_page, offset])
        else:
            # No search - use regular query
            where_clauses = []
            params = []
            
            if camera:
                where_clauses.append("camera_name = ?")
                params.append(camera)
            
            if date:
                where_clauses.append("DATE(timestamp) = ?")
                params.append(date)
            
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
        
        transcriptions = []
        for row in rows:
            segments = None
            try:
                raw_segments = row["segments"]
                if raw_segments:
                    segments = json.loads(raw_segments)
            except (KeyError, json.JSONDecodeError):
                segments = None
            
            transcriptions.append({
                "id": row["id"],
                "event_id": row["event_id"],
                "camera_name": row["camera_name"],
                "timestamp": row["timestamp"],
                "transcription": row["transcription"],
                "segments": segments,
                "language": row["language"],
                "duration_seconds": row["duration_seconds"],
                "status": row["status"],
                "audio_file": row["audio_file"]
            })
        
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


# Available Whisper models for selection
# Note: speaches uses faster-whisper which needs CTranslate2-format models
AVAILABLE_MODELS = [
    {"id": "Systran/faster-whisper-large-v3", "name": "Large V3 (Best accuracy)", "size": "~3GB"},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "name": "Large V3 Turbo (6x faster)", "size": "~1.6GB"},
    {"id": "Systran/faster-whisper-medium", "name": "Medium (Balanced)", "size": "~1.5GB"},
    {"id": "Systran/faster-whisper-small", "name": "Small (Fast)", "size": "~500MB"},
]

# Available languages
AVAILABLE_LANGUAGES = [
    {"code": "da", "name": "Danish"},
    {"code": "en", "name": "English"},
    {"code": "de", "name": "German"},
    {"code": "sv", "name": "Swedish"},
    {"code": "no", "name": "Norwegian"},
    {"code": "auto", "name": "Auto-detect"},
]


@app.get("/api/settings")
async def api_get_settings():
    """Get all configurable settings."""
    settings = get_settings()
    return {
        "settings": settings,
        "available_models": AVAILABLE_MODELS,
        "available_languages": AVAILABLE_LANGUAGES
    }


@app.put("/api/settings")
async def api_update_settings(request: Request):
    """Update settings."""
    global protect_client
    
    try:
        data = await request.json()
        
        # Validate and save each setting
        allowed_keys = ['whisper_model', 'language', 'buffer_before', 'buffer_after', 'vad_filter', 'beam_size', 'protect_host']
        
        updated = []
        protect_host_changed = False
        
        for key, value in data.items():
            if key in allowed_keys:
                # Validate specific settings
                if key in ['buffer_before', 'buffer_after', 'beam_size']:
                    try:
                        int_val = int(value)
                        # buffer_before: 1-60s (lead-in context)
                        # buffer_after: 1-600s (10 min for longer conversations)
                        if key == 'buffer_before' and (int_val < 1 or int_val > 60):
                            raise ValueError(f"buffer_before must be between 1 and 60 seconds")
                        if key == 'buffer_after' and (int_val < 1 or int_val > 600):
                            raise ValueError(f"buffer_after must be between 1 and 600 seconds (10 min)")
                        if key == 'beam_size' and (int_val < 1 or int_val > 10):
                            raise ValueError("beam_size must be between 1 and 10")
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e))
                
                if key == 'vad_filter':
                    value = 'true' if value in [True, 'true', '1', 1] else 'false'
                
                if key == 'protect_host':
                    # Clean up the host - remove trailing slashes, protocol if present
                    value = str(value).strip().rstrip('/')
                    if value.startswith('https://'):
                        value = value[8:]
                    if value.startswith('http://'):
                        value = value[7:]
                    protect_host_changed = True
                
                save_setting(key, str(value))
                updated.append(key)
        
        # If protect_host changed, invalidate the client so it reconnects with new host
        if protect_host_changed:
            protect_client = None
            logger.info("Protect host changed, client will reconnect on next request")
        
        return {"status": "updated", "updated_keys": updated, "settings": get_settings()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating settings: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/settings/test-whisper")
async def test_whisper_connection():
    """Test connection to Whisper server and get available models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{WHISPER_URL}/v1/models")
            
            if response.status_code == 200:
                models = response.json()
                return {
                    "status": "connected",
                    "whisper_url": WHISPER_URL,
                    "models": models
                }
            else:
                return {
                    "status": "error",
                    "message": f"Whisper returned status {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/settings/test-protect")
async def test_protect_connection():
    """Test connection to UniFi Protect NVR."""
    try:
        host = get_protect_host()
        if not host:
            return {
                "status": "error",
                "message": "Protect host not configured"
            }
        
        client = await get_protect_client(force_reconnect=True)
        nvr = client.bootstrap.nvr
        cameras = list(client.bootstrap.cameras.values())
        
        return {
            "status": "connected",
            "host": host,
            "nvr_name": nvr.name,
            "nvr_version": nvr.version,
            "camera_count": len(cameras),
            "cameras": [{"id": c.id, "name": c.name} for c in cameras]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/sync")
async def sync_speech_events(
    hours: int = Query(default=24, ge=1, le=720)  # 1 hour to 30 days
):
    """
    Sync speech events from UniFi Protect NVR.
    Fetches recent speech detection events and queues any missing ones for transcription.
    Skips live/ongoing events (where end is None) to avoid file errors.
    """
    try:
        host = get_protect_host()
        if not host:
            raise HTTPException(status_code=400, detail="Protect host not configured. Set it in Settings.")
        
        client = await get_protect_client()
        
        # Calculate time range
        end_time = datetime.now(tz=LOCAL_TZ)
        start_time = end_time - timedelta(hours=hours)
        
        logger.info(f"Syncing speech events from {start_time.isoformat()} to {end_time.isoformat()}")
        
        # Get existing event IDs to avoid duplicates
        conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("SELECT event_id FROM transcriptions")
        existing_events = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        # Get settings for language
        settings = get_settings()
        language = settings.get('language', 'da')
        
        events_found = 0
        events_queued = 0
        events_skipped = 0
        events_live_skipped = 0
        errors = []
        speech_events_found = 0
        all_smart_types = set()
        all_event_types = set()
        
        # Try to get events using the get_events method
        try:
            # The uiprotect library's get_events method
            events = await client.get_events(
                start=start_time,
                end=end_time,
            )
            
            for event in events:
                events_found += 1
                
                event_type = getattr(event, 'type', None)
                if event_type:
                    all_event_types.add(str(event_type))
                
                # Check if this is a speech/smart detect event
                smart_detect_types = getattr(event, 'smart_detect_types', None)
                if smart_detect_types is None:
                    continue
                
                # Collect all types and convert to list of strings for comparison
                smart_types_str = []
                for t in smart_detect_types:
                    # Handle both enum values and strings
                    if hasattr(t, 'value'):
                        type_str = str(t.value).lower()
                    elif hasattr(t, 'name'):
                        type_str = str(t.name).lower()
                    else:
                        type_str = str(t).lower()
                    smart_types_str.append(type_str)
                    all_smart_types.add(type_str)
                
                # Only process speech events - UniFi calls it 'alrmspeak'
                is_speech = any(s in ['alrmspeak', 'speech', 'speechdetect'] for s in smart_types_str)
                if not is_speech:
                    continue
                
                speech_events_found += 1
                
                # CRUCIAL: Skip live/ongoing events (end is None) to avoid file errors
                event_end = getattr(event, 'end', None)
                if event_end is None:
                    events_live_skipped += 1
                    logger.debug(f"Skipping live event {event.id} (end is None)")
                    continue
                
                event_id = str(event.id)
                
                if event_id in existing_events:
                    events_skipped += 1
                    continue
                
                # Get camera info
                camera_id = getattr(event, 'camera_id', None)
                if not camera_id:
                    camera = getattr(event, 'camera', None)
                    if camera:
                        camera_id = camera.id
                
                if not camera_id:
                    logger.warning(f"Event {event_id} has no camera_id, skipping")
                    continue
                
                # Get timestamp from event
                event_time = getattr(event, 'start', None)
                if not event_time:
                    logger.warning(f"Event {event_id} has no start time, skipping")
                    continue
                
                timestamp_ms = int(event_time.timestamp() * 1000)
                
                camera = client.bootstrap.cameras.get(camera_id)
                camera_name = camera.name if camera else f"Unknown ({camera_id})"
                
                logger.info(f"Queuing missing event {event_id} from {camera_name} at {event_time}")
                
                # Queue for processing (worker will pick it up)
                if queue_transcription(event_id, str(camera_id), camera_name, timestamp_ms, language):
                    events_queued += 1
                    existing_events.add(event_id)  # Track to avoid duplicate queuing
                
        except AttributeError as e:
            error_msg = f"API method not available: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error fetching events: {e}"
            logger.exception(error_msg)
            errors.append(error_msg)
        
        # Log all unique types found for debugging
        logger.info(f"DEBUG All event types found: {all_event_types}")
        logger.info(f"DEBUG All smart detect types found: {all_smart_types}")
        logger.info(f"DEBUG Speech events found: {speech_events_found}, live skipped: {events_live_skipped}")
        
        result = {
            "status": "completed",
            "hours_searched": hours,
            "events_found": events_found,
            "speech_events_found": speech_events_found,
            "events_queued": events_queued,
            "events_skipped": events_skipped,
            "events_live_skipped": events_live_skipped,
            "message": f"Queued {events_queued} new events for transcription",
            "debug_smart_types": list(all_smart_types),  # Include in response for easy viewing
            "debug_event_types": list(all_event_types)
        }
        
        if errors:
            result["errors"] = errors
            result["status"] = "completed_with_errors"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error syncing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription(transcription_id: int):
    """Delete a transcription and its audio file."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get the transcription first to find the audio file
        cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        # Delete the audio file if it exists
        if row["audio_file"]:
            audio_path = Path(AUDIO_PATH) / row["audio_file"]
            try:
                if audio_path.exists():
                    audio_path.unlink()
                    logger.info(f"Deleted audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Could not delete audio file {audio_path}: {e}")
                # Continue with database deletion even if file deletion fails
        
        # Delete the database record
        cursor.execute("DELETE FROM transcriptions WHERE id = ?", (transcription_id,))
        conn.commit()
        
        logger.info(f"Deleted transcription {transcription_id}")
        return {"status": "deleted", "id": transcription_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/transcriptions/{transcription_id}/srt")
async def download_srt(transcription_id: int):
    """Download transcription as SRT subtitle file."""
    from fastapi.responses import PlainTextResponse
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        segments = []
        if row["segments"]:
            try:
                segments = json.loads(row["segments"])
            except json.JSONDecodeError:
                pass
        
        if not segments:
            # Fall back to single segment with full text
            segments = [{
                "start": 0,
                "end": row["duration_seconds"] or 10,
                "text": row["transcription"] or ""
            }]
        
        # Generate SRT format
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", start + 5)
            text = seg.get("text", "").strip()
            
            if text:
                start_srt = format_srt_time(start)
                end_srt = format_srt_time(end)
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_srt} --> {end_srt}")
                srt_lines.append(text)
                srt_lines.append("")
        
        srt_content = "\n".join(srt_lines)
        
        # Create filename from camera and timestamp
        camera = row["camera_name"] or "unknown"
        timestamp = row["timestamp"] or "unknown"
        filename = f"{camera}_{timestamp}.srt".replace(" ", "_").replace(":", "-")
        
        return PlainTextResponse(
            content=srt_content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
        
    finally:
        conn.close()


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@app.post("/api/database/reset")
async def reset_database():
    """
    Factory reset: Delete all transcriptions and audio files, but keep settings.
    This is a destructive operation that cannot be undone.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Count records being deleted
        cursor.execute("SELECT COUNT(*) FROM transcriptions")
        transcription_count = cursor.fetchone()[0]
        
        # Delete all transcriptions (FTS table will be updated via trigger)
        cursor.execute("DELETE FROM transcriptions")
        conn.commit()
        
        # Vacuum the database to reclaim space
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()
        
        # Delete all audio files
        audio_files_deleted = 0
        audio_dir = Path(AUDIO_PATH)
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.wav"):
                try:
                    audio_file.unlink()
                    audio_files_deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete audio file {audio_file}: {e}")
        
        logger.info(f"Database reset: deleted {transcription_count} transcriptions and {audio_files_deleted} audio files")
        
        return {
            "status": "success",
            "message": "Database reset successfully",
            "transcriptions_deleted": transcription_count,
            "audio_files_deleted": audio_files_deleted
        }
        
    except Exception as e:
        logger.exception(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcriptions/{transcription_id}/retry")
async def retry_transcription(transcription_id: int):
    """Retry a transcription - re-fetches audio and re-transcribes."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get the original transcription record
        cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (transcription_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        event_id = row["event_id"]
        camera_id = row["camera_id"]
        camera_name = row["camera_name"]
        timestamp_str = row["timestamp"]
        
        # Parse the timestamp back to milliseconds
        try:
            # Handle ISO format timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            timestamp_ms = int(dt.timestamp() * 1000)
        except Exception as e:
            logger.error(f"Failed to parse timestamp {timestamp_str}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {timestamp_str}")
        
        # Delete the old audio file if it exists
        if row["audio_file"]:
            old_audio_path = Path(AUDIO_PATH) / row["audio_file"]
            if old_audio_path.exists():
                old_audio_path.unlink()
        
        # Delete the old record (so we can re-insert with pending status)
        cursor.execute("DELETE FROM transcriptions WHERE id = ?", (transcription_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Retrying transcription {transcription_id} (event {event_id})")
        
        # Get settings for language
        settings = get_settings()
        language = settings.get('language', 'da')
        
        # Queue for processing (worker will pick it up)
        queue_transcription(event_id, camera_id, camera_name, timestamp_ms, language)
        
        return {
            "status": "queued",
            "id": transcription_id,
            "event_id": event_id,
            "message": "Transcription retry queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        logger.exception(f"Error retrying transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
