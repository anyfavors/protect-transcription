"""
Sync historical speech events from the UniFi Protect NVR.
Shared by the manual /api/sync route and the periodic auto-sync worker.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from uiprotect.data import EventType, SmartDetectObjectType

from app.config import LOCAL_TZ
from app.database import get_connection, get_settings
from app.protect import get_protect_client, get_protect_host
from app.worker import queue_transcription

logger = logging.getLogger(__name__)

# Module-level lock — guarantees that manual and auto-sync cannot run concurrently
# (which would double-fetch the same Protect events and waste API quota).
_sync_lock = asyncio.Lock()


class SyncError(Exception):
    """Raised when sync cannot proceed (e.g. Protect unreachable)."""


async def run_sync(hours: int) -> dict:
    """Serialize sync runs through a single async lock and delegate."""
    if _sync_lock.locked():
        logger.info("Sync already in progress; queued behind running sync")
    async with _sync_lock:
        return await _run_sync_impl(hours)


async def _run_sync_impl(hours: int) -> dict:
    """
    Fetch speech events from Protect for the last *hours* hours and queue any missing ones.
    Returns a result dict (same shape as the /api/sync response).
    Raises SyncError if Protect is not configured or unreachable.
    """
    host = get_protect_host()
    if not host:
        raise SyncError("Protect host not configured")

    try:
        client = await get_protect_client()
    except Exception as exc:
        raise SyncError(f"Cannot connect to Protect: {exc}") from exc

    end_time = datetime.now(tz=LOCAL_TZ)
    start_time = end_time - timedelta(hours=hours)
    logger.info("Syncing speech events %s → %s", start_time.isoformat(), end_time.isoformat())

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT event_id FROM transcriptions")
    existing_events = {row[0] for row in cursor.fetchall()}
    conn.close()

    settings = get_settings()
    language = settings.get("language", "da")

    events_found = events_queued = events_skipped = events_live_skipped = speech_events_found = 0
    all_smart_types: set[str] = set()
    all_event_types: set[str] = set()
    errors: list[str] = []

    try:
        # Filter on the NVR: only speech (alrmSpeak) smart-audio events. Raw dicts
        # (ms timestamps, camera id) avoid parsing every event into models.
        events = await client.get_events_raw(
            start=start_time,
            end=end_time,
            types=[EventType.SMART_AUDIO_DETECT],
            smart_detect_types=[SmartDetectObjectType.SPEAK],
        )

        for event in events:
            events_found += 1
            all_event_types.add(str(event.get("type")))
            all_smart_types.update(str(t).lower() for t in event.get("smartDetectTypes") or [])
            speech_events_found += 1

            if event.get("end") is None:
                events_live_skipped += 1
                continue

            event_id = str(event["id"])
            if event_id in existing_events:
                events_skipped += 1
                continue

            camera_id = event.get("camera")
            timestamp_ms = event.get("start")
            if not camera_id or not timestamp_ms:
                continue

            camera = client.bootstrap.cameras.get(camera_id)
            camera_name: str = (
                (camera.name or f"Unknown ({camera_id})") if camera else f"Unknown ({camera_id})"
            )

            if queue_transcription(
                event_id, str(camera_id), camera_name, int(timestamp_ms), language
            ):
                events_queued += 1
                existing_events.add(event_id)

    except AttributeError as exc:
        errors.append(f"API method not available: {exc}")
        logger.error(errors[-1])
    except Exception as exc:
        errors.append(f"Error fetching events: {exc}")
        logger.exception(errors[-1])

    logger.info(
        "Sync done: found=%d speech=%d queued=%d skipped=%d live_skipped=%d",
        events_found,
        speech_events_found,
        events_queued,
        events_skipped,
        events_live_skipped,
    )

    result: dict = {
        "status": "completed",
        "hours_searched": hours,
        "events_found": events_found,
        "speech_events_found": speech_events_found,
        "events_queued": events_queued,
        "events_skipped": events_skipped,
        "events_live_skipped": events_live_skipped,
        "message": f"Queued {events_queued} new events for transcription",
        "debug_smart_types": list(all_smart_types),
        "debug_event_types": list(all_event_types),
    }
    if errors:
        result["errors"] = errors
        result["status"] = "completed_with_errors"

    return result
