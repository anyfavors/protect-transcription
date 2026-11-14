"""
Background transcription worker, audio compression worker, and queue helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

from app.config import AUDIO_PATH, DATABASE_PATH, LOCAL_TZ
from app.database import get_settings, reset_stuck_processing
from app.metrics import inc, set_gauge
from app.protect import get_protect_client
from app.transcription import (
    _extract_audio,
    compute_audio_rms,
    fetch_audio_clip,
    save_audio_file,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

# Rows the worker may pick up now (deferred / backing-off rows have a future next_attempt_at).
READY_SQL = (
    "status = 'pending' AND "
    "(next_attempt_at IS NULL OR next_attempt_at <= strftime('%Y-%m-%dT%H:%M:%S', 'now'))"
)


def _open_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def queue_transcription(
    event_id: str,
    camera_id: str,
    camera_name: str,
    timestamp_ms: int,
    language: str = "da",
) -> bool:
    """
    Insert a pending transcription row.
    Returns True if queued, False if the event already exists.
    """
    conn = _open_conn()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM transcriptions WHERE event_id = ?", (event_id,))
        if cursor.fetchone():
            logger.debug("Event %s already exists, skipping", event_id)
            return False

        event_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=LOCAL_TZ)
        cursor.execute(
            """
            INSERT INTO transcriptions
                (event_id, camera_id, camera_name, timestamp, status, language)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (event_id, camera_id, camera_name, event_time.isoformat(), language),
        )
        conn.commit()
        inc("events_queued_total")
        logger.info("Queued transcription for event %s from %s", event_id, camera_name)
        return True

    except sqlite3.IntegrityError:
        logger.debug("Event %s already exists (integrity error)", event_id)
        return False
    finally:
        conn.close()


_PROTECT_EVENT_ID = re.compile(r"^[0-9a-f]{24}$")
# Event length assumed when the event can't be looked up on the NVR.
_FALLBACK_EVENT_SECONDS = 60
# Stop waiting for an event that never reports an end.
_MAX_EVENT_WAIT = timedelta(hours=1)
# ponytail: hard cap on clip length; raise if long conversations get cut off.
_MAX_CLIP_SECONDS = 900
# Give the NVR time to finish writing footage before exporting it.
_RECORDING_SETTLE_SECONDS = 15
# Transient failures (NVR / speaches unreachable) are retried with backoff.
_MAX_ATTEMPTS = 5
_RETRY_BASE_MINUTES = 5
# Failed NVR downloads are only retried for events younger than this.
_RETRY_FETCH_WINDOW = timedelta(days=1)


def _parse_local(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=LOCAL_TZ)


def _utc_stamp(dt: datetime) -> str:
    """Format for next_attempt_at; compares as text with strftime(..., 'now')."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


async def _event_window(row: dict, event_time: datetime) -> tuple[datetime, datetime | None]:
    """(start, end) of the speech event. end is None while the event is still ongoing."""
    if row.get("event_end"):
        return event_time, _parse_local(row["event_end"])
    if _PROTECT_EVENT_ID.match(row.get("event_id") or ""):
        try:
            client = await get_protect_client()
            event = await client.get_event(row["event_id"])
            end = event.end.astimezone(LOCAL_TZ) if event.end else None
            return event.start.astimezone(LOCAL_TZ), end
        except Exception as exc:
            logger.warning("Could not look up event %s: %s", row["event_id"], exc)
    return event_time, event_time + timedelta(seconds=_FALLBACK_EVENT_SECONDS)


async def process_pending_transcription(row: dict) -> None:
    """Fetch audio, transcribe, and persist the result for one pending row."""
    event_id = row["event_id"]
    camera_id = row["camera_id"]
    camera_name = row["camera_name"]
    timestamp_str = row["timestamp"]
    record_id = row["id"]
    attempts = int(row.get("attempts") or 0)

    settings = get_settings()
    buffer_before = int(settings.get("buffer_before", "5"))
    buffer_after = int(settings.get("buffer_after", "10"))

    try:
        event_time = _parse_local(timestamp_str)
    except Exception as exc:
        _set(record_id, status="error", transcription=f"Invalid timestamp: {timestamp_str} ({exc})")
        return

    existing_audio_file: str | None = row.get("audio_file")
    cached_path = Path(AUDIO_PATH) / existing_audio_file if existing_audio_file else None
    has_cache = bool(cached_path and cached_path.exists())
    need_fetch = not has_cache or bool(row.get("refetch"))

    audio_data: bytes | None = None
    audio_filename: str | None = None
    if need_fetch:
        now = datetime.now(tz=LOCAL_TZ)
        start, end = await _event_window(row, event_time)
        if end is None:
            if now - start < _MAX_EVENT_WAIT:
                logger.info("Event %s still in progress; checking again in 60s", event_id)
                _set(
                    record_id,
                    status="pending",
                    next_attempt_at=_utc_stamp(now + timedelta(seconds=60)),
                )
                return
            end = start + timedelta(seconds=_FALLBACK_EVENT_SECONDS)

        clip_start = start - timedelta(seconds=buffer_before)
        clip_end = min(
            end + timedelta(seconds=buffer_after),
            clip_start + timedelta(seconds=_MAX_CLIP_SECONDS),
        )
        ready_at = clip_end + timedelta(seconds=_RECORDING_SETTLE_SECONDS)
        if ready_at > now:
            logger.info("Event %s: waiting until %s for footage", event_id, ready_at.isoformat())
            _set(record_id, status="pending", next_attempt_at=_utc_stamp(ready_at))
            return

        logger.info(
            "Processing event %s from %s: clip %s → %s (%.0fs)",
            event_id,
            camera_name,
            clip_start.isoformat(),
            clip_end.isoformat(),
            (clip_end - clip_start).total_seconds(),
        )
        fetched = await fetch_audio_clip(camera_id, clip_start, clip_end)
        if fetched:
            audio_data = fetched
            audio_filename = save_audio_file(fetched, start, camera_name)
            if cached_path and has_cache and audio_filename != existing_audio_file:
                cached_path.unlink(missing_ok=True)
            # Record the real event bounds (webhook timestamps are trigger times).
            _set(record_id, timestamp=start.isoformat(), event_end=end.isoformat())
            timestamp_str = start.isoformat()
        elif not has_cache:
            # Retrying only helps for recent events (NVR blip); old footage is gone for good.
            if now - start > _RETRY_FETCH_WINDOW:
                attempts = _MAX_ATTEMPTS
            _retry_or_fail(record_id, attempts, "Failed to fetch audio", None)
            await _broadcast_update(record_id, "error", camera_name, timestamp_str)
            return
        else:
            logger.info("Re-fetch failed for event %s; using cached audio", event_id)

    if audio_data is None:
        assert cached_path is not None
        logger.info("Reusing cached audio file: %s", cached_path.name)
        audio_data = cached_path.read_bytes()
        audio_filename = existing_audio_file
        if not audio_data.startswith(b"RIFF"):
            # Older clips are compressed to Opus/OGG; decode back to 16 kHz WAV.
            audio_data = await _extract_audio(audio_data)
            if not audio_data:
                _retry_or_fail(
                    record_id, attempts, "Failed to decode cached audio", existing_audio_file
                )
                return

    duration = len(audio_data) / (16000 * 2)  # 16 kHz, 16-bit

    # ── Noise / silence pre-filter ──────────────────────────────
    min_energy = float(settings.get("min_audio_energy", "0.005"))
    if min_energy > 0:
        rms = compute_audio_rms(audio_data)
        if rms < min_energy:
            logger.info("Audio filtered (RMS %.5f < %.5f) for event %s", rms, min_energy, event_id)
            inc("transcriptions_filtered_total")
            _finish(
                record_id,
                "filtered",
                f"Silence/noise detected (RMS energy {rms:.5f} below threshold {min_energy})",
                audio_filename,
                duration,
            )
            await _broadcast_update(record_id, "filtered", camera_name, timestamp_str)
            return

    result = await transcribe_audio(audio_data)

    if "error" in result:
        final_status = _retry_or_fail(
            record_id, attempts, f"Transcription error: {result['error']}", audio_filename
        )
    elif result.get("hallucination"):
        inc("transcriptions_filtered_total")
        _finish(
            record_id,
            "filtered",
            f"Hallucination discarded: {str(result.get('raw_text') or '')[:300]}",
            audio_filename,
            duration,
        )
        final_status = "filtered"
    elif result.get("empty"):
        inc("transcriptions_filtered_total")
        _finish(
            record_id,
            "filtered",
            "Empty transcription (no speech detected by Whisper)",
            audio_filename,
            duration,
        )
        final_status = "filtered"
    else:
        segments = result.get("segments", [])
        _finish(
            record_id,
            "completed",
            result.get("text", ""),
            audio_filename,
            duration,
            segments=json.dumps(segments) if segments else None,
            language=result.get("language", "da"),
            confidence=result.get("confidence", 0),
        )
        final_status = "completed"
        # Alerts only for first-time results of recent events, never for re-runs.
        if not row.get("processed_at") and not row.get("refetch"):
            from app.alerts import check_alerts

            await check_alerts(record_id, camera_name, timestamp_str, result.get("text", ""))

    logger.info("Finished event %s from %s: %s", event_id, camera_name, final_status)
    await _broadcast_update(record_id, final_status, camera_name, timestamp_str)


def _set(record_id: int, **fields: object) -> None:
    """UPDATE the given columns (names are code-controlled, never user input)."""
    cols = ", ".join(f"{k}=?" for k in fields)
    conn = _open_conn()
    try:
        conn.execute(f"UPDATE transcriptions SET {cols} WHERE id=?", (*fields.values(), record_id))
        conn.commit()
    finally:
        conn.close()


def _finish(
    record_id: int,
    status: str,
    text: str,
    audio_filename: str | None,
    duration: float,
    **extra: object,
) -> None:
    """Store a final result and clear retry/refetch bookkeeping.

    A hand-corrected row keeps the user's text and status; the new machine
    output goes to original_transcription instead.
    """
    conn = _open_conn()
    try:
        edited = conn.execute(
            "SELECT edited_at IS NOT NULL FROM transcriptions WHERE id=?", (record_id,)
        ).fetchone()
    finally:
        conn.close()
    if edited and edited[0]:
        extra = {k: v for k, v in extra.items() if k != "segments"}
        _set(
            record_id,
            status="completed",
            original_transcription=text,
            audio_file=audio_filename,
            duration_seconds=duration,
            attempts=0,
            next_attempt_at=None,
            refetch=0,
            priority=0,
            processed_at=_utc_stamp(datetime.now(tz=LOCAL_TZ)),
            **extra,
        )
        return
    _set(
        record_id,
        status=status,
        transcription=text,
        audio_file=audio_filename,
        duration_seconds=duration,
        attempts=0,
        next_attempt_at=None,
        refetch=0,
        priority=0,
        processed_at=_utc_stamp(datetime.now(tz=LOCAL_TZ)),
        **extra,
    )


def _retry_or_fail(record_id: int, attempts: int, message: str, audio_filename: str | None) -> str:
    """Re-queue a transient failure with exponential backoff, or mark it as error."""
    attempts += 1
    if attempts < _MAX_ATTEMPTS:
        delay = timedelta(minutes=_RETRY_BASE_MINUTES * 2 ** (attempts - 1))
        logger.warning("%s (attempt %d); retrying in %s", message, attempts, delay)
        inc("transcription_retries_total")
        fields: dict[str, object] = {
            "status": "pending",
            "attempts": attempts,
            "next_attempt_at": _utc_stamp(datetime.now(tz=LOCAL_TZ) + delay),
        }
        if audio_filename:
            fields["audio_file"] = audio_filename
        _set(record_id, **fields)
        return "pending"
    _set(
        record_id,
        status="error",
        transcription=message[:1000],
        audio_file=audio_filename,
        attempts=attempts,
        next_attempt_at=None,
        refetch=0,
        priority=0,
        processed_at=_utc_stamp(datetime.now(tz=LOCAL_TZ)),
    )
    return "error"


async def _claim_next_pending() -> dict | None:
    """Atomically claim the next ready pending row and return it as a dict.

    Order: rows re-queued by hand, then new events (no audio yet), so a mass
    re-transcription doesn't delay live ones; the backlog is worked newest-first.
    """
    conn = _open_conn()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute(
            f"""
            SELECT id, event_id, camera_id, camera_name, timestamp, language, audio_file,
                   event_end, attempts, refetch, processed_at
            FROM transcriptions
            WHERE {READY_SQL}
            ORDER BY priority DESC, (audio_file IS NOT NULL), timestamp DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        cur.execute(
            "UPDATE transcriptions SET status='processing' WHERE id=? AND status='pending'",
            (row["id"],),
        )
        if cur.rowcount == 0:
            conn.commit()
            return None
        conn.commit()
        return dict(row)
    finally:
        conn.close()


async def transcription_worker() -> None:
    """Infinite loop: atomically claim one pending row and process it."""
    while True:
        try:
            row = await _claim_next_pending()
            if row:
                await process_pending_transcription(row)
                set_gauge("queue_pending_rows", _count_pending())
                await asyncio.sleep(1)
            else:
                set_gauge("queue_pending_rows", 0)
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Error in transcription worker: %s", exc)
            inc("worker_loop_errors_total")
            await asyncio.sleep(10)


def _count_pending() -> int:
    conn = _open_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transcriptions WHERE status='pending'")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


async def _broadcast_update(record_id: int, status: str, camera_name: str, timestamp: str) -> None:
    """Push a transcription status change to all connected WebSocket clients."""
    from app.broadcast import broadcast

    await broadcast(
        {
            "type": "transcription_update",
            "id": record_id,
            "status": status,
            "camera_name": camera_name,
            "timestamp": timestamp,
        }
    )


# ---------------------------------------------------------------------------
# Periodic workers
# ---------------------------------------------------------------------------
async def audio_compression_worker() -> None:
    """
    Periodically compress old WAV audio files to Opus/OGG.

    Runs hourly. Controlled by the ``audio_compression_days`` setting:
    WAV files referenced by transcriptions older than N days are converted
    to OGG (libopus @ 32 kbps), the original is deleted, and the DB is updated.
    Set to 0 to disable.
    """
    while True:
        try:
            settings = get_settings()
            days = int(settings.get("audio_compression_days", "7"))
            if days > 0:
                await _run_audio_compression(days)
            await asyncio.sleep(3600)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Audio compression worker error: %s", exc)
            await asyncio.sleep(60)


async def _run_audio_compression(days: int) -> None:
    """Single compression pass — extracted so it's testable and re-runnable."""
    conn = _open_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, audio_file FROM transcriptions "
            "WHERE audio_file LIKE '%.wav' AND timestamp < datetime('now', ?)",
            (f"-{days} days",),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return

    compressed = 0
    for row_id, audio_file in rows:
        wav_path = Path(AUDIO_PATH) / audio_file
        if not wav_path.exists():
            continue

        ogg_name = audio_file.rsplit(".", 1)[0] + ".ogg"
        ogg_path = Path(AUDIO_PATH) / ogg_name

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                "-vbr",
                "on",
                str(ogg_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, _stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                rc = proc.returncode
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("ffmpeg timeout compressing %s", audio_file)
                ogg_path.unlink(missing_ok=True)
                continue
        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH; cannot compress audio")
            return

        if rc == 0 and ogg_path.exists() and ogg_path.stat().st_size > 0:
            wav_path.unlink(missing_ok=True)
            conn = _open_conn()
            try:
                conn.execute(
                    "UPDATE transcriptions SET audio_file=? WHERE id=?",
                    (ogg_name, row_id),
                )
                conn.commit()
            finally:
                conn.close()
            compressed += 1
        else:
            ogg_path.unlink(missing_ok=True)
            logger.warning("Failed to compress %s (rc=%s)", audio_file, rc)

    if compressed:
        inc("audio_files_compressed_total", compressed)
        logger.info("Compressed %d audio files to Opus/OGG", compressed)


async def auto_sync_worker() -> None:
    """
    Periodically pull recent speech events from the Protect NVR.
    Webhook delivery is the realtime path; this is the safety net.

    Cadence and lookback are read live from settings (auto_sync_interval_minutes,
    auto_sync_hours), so changes take effect on the next iteration.
    """
    await asyncio.sleep(60)  # Let Protect client init first

    while True:
        try:
            from app.sync_service import SyncError, run_sync

            settings = get_settings()
            try:
                hours = int(settings.get("auto_sync_hours", "24"))
            except ValueError:
                hours = 24
            try:
                interval_minutes = int(settings.get("auto_sync_interval_minutes", "60"))
            except ValueError:
                interval_minutes = 60
            interval_minutes = max(5, interval_minutes)

            try:
                result = await run_sync(hours=hours)
                inc("auto_sync_runs_total")
                logger.info(
                    "Auto-sync: queued=%d skipped=%d live_skipped=%d",
                    result.get("events_queued", 0),
                    result.get("events_skipped", 0),
                    result.get("events_live_skipped", 0),
                )
            except SyncError as exc:
                logger.warning("Auto-sync skipped: %s", exc)
            except Exception as exc:
                inc("auto_sync_errors_total")
                logger.exception("Auto-sync error: %s", exc)

            await asyncio.sleep(interval_minutes * 60)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Auto-sync worker loop error: %s", exc)
            await asyncio.sleep(300)


def _parse_hhmm(value: str, default: time) -> time:
    try:
        h, m = value.split(":", 1)
        return time(hour=int(h), minute=int(m))
    except Exception:
        return default


async def _generate_summary_safe(period: str, key: str) -> bool:
    """Generate one summary; log and swallow failures (no transcripts / Ollama down)."""
    from app.summaries import generate_summary

    try:
        await generate_summary(period, key)
        inc("summaries_generated_total")
        return True
    except Exception as exc:
        logger.info("Auto-summary skipped for %s %s: %s", period, key, exc)
        return False


async def run_rollups(include_current: bool = True) -> int:
    """
    Regenerate every summary that is missing or whose inputs changed.

    Levels run in dependency order (daily → weekly/monthly → yearly), so a
    changed transcript refreshes its day, and the new day summary in turn
    makes its week/month stale in the same pass, and so on up to the year.
    Periods containing "now" are only touched when *include_current* is set
    (the nightly run), so the running day/week/month/year isn't rebuilt
    after every new clip.
    """
    from app.summaries import PERIODS, current_keys, get_summaries

    current = current_keys()
    generated = 0
    for period in PERIODS:
        try:
            items = get_summaries(period, limit=None)["items"]
        except Exception as exc:
            logger.warning("Rollup: failed to read %s summaries: %s", period, exc)
            continue
        for item in reversed(items):  # oldest first
            key = item["period_key"]
            if key == current[period] and not include_current:
                continue
            if item.get("summary") and not item.get("stale"):
                continue
            logger.info("Rollup: generating %s summary for %s", period, key)
            if await _generate_summary_safe(period, key):
                generated += 1
    return generated


def _queue_busy() -> bool:
    """True while transcriptions are being processed or are ready to be."""
    conn = _open_conn()
    try:
        cur = conn.execute(
            f"SELECT EXISTS(SELECT 1 FROM transcriptions WHERE status='processing' OR ({READY_SQL}))"
        )
        return bool(cur.fetchone()[0])
    finally:
        conn.close()


_ROLLUP_CHECK_SECONDS = 300
# Wait until the transcription queue has been empty this long before
# summarising, so a mass re-transcription settles before anything is rebuilt.
_ROLLUP_QUIET = timedelta(minutes=30)


async def auto_summary_worker() -> None:
    """
    Keep summaries in sync with transcriptions.

    Every few minutes: if the transcription queue has been idle for
    _ROLLUP_QUIET, regenerate summaries whose source content changed (closed
    periods). Once a day after ``auto_summary_time`` the current periods are
    included too. Nothing is regenerated unless its inputs actually changed.
    """
    from app.summaries import backfill_source_hashes

    try:
        backfill_source_hashes()
    except Exception as exc:
        logger.exception("Summary hash backfill failed: %s", exc)

    quiet_since: datetime | None = None
    last_nightly = None
    while True:
        try:
            await asyncio.sleep(_ROLLUP_CHECK_SECONDS)
            now = datetime.now(tz=LOCAL_TZ)
            if _queue_busy():
                quiet_since = None
                continue
            quiet_since = quiet_since or now
            if now - quiet_since < _ROLLUP_QUIET:
                continue

            target = _parse_hhmm(get_settings().get("auto_summary_time", "23:55"), time(23, 55))
            nightly = now.time() >= target and last_nightly != now.date()
            count = await run_rollups(include_current=nightly)
            if nightly:
                last_nightly = now.date()
            if count:
                logger.info("Rollup pass generated %d summaries (nightly=%s)", count, nightly)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Auto-summary worker loop error: %s", exc)
            await asyncio.sleep(300)


async def stuck_processing_recovery_worker() -> None:
    """
    Periodically reclaim rows stuck in 'processing'.

    This handles two failure modes:
    * Pod was killed mid-transcription (SIGTERM during process_pending_transcription).
    * Worker process died but the row was never reset.

    Runs every 5 minutes, controlled by `processing_timeout_minutes` setting.
    """
    while True:
        try:
            settings = get_settings()
            try:
                timeout_min = int(settings.get("processing_timeout_minutes", "10"))
            except ValueError:
                timeout_min = 10
            count = reset_stuck_processing(timeout_min)
            if count:
                inc("stuck_processing_resets_total", count)
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Stuck-processing recovery error: %s", exc)
            await asyncio.sleep(300)


async def protect_health_worker() -> None:
    """
    Periodically refresh the Protect bootstrap so dead TCP connections are
    detected proactively (instead of waiting for the next webhook/sync to fail).
    """
    while True:
        try:
            await asyncio.sleep(300)
            from app.protect import get_protect_client

            try:
                client = await get_protect_client()
                await client.update()
                set_gauge("protect_connected", 1)
            except Exception as exc:
                set_gauge("protect_connected", 0)
                logger.warning("Protect health check failed: %s", exc)
                inc("protect_health_failures_total")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Protect health worker error: %s", exc)
            await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# Legacy helper kept for webhook / retry compatibility
# ---------------------------------------------------------------------------
async def process_speech_event(
    event_id: str,
    camera_id: str,
    timestamp_ms: int,
    skip_wait: bool = False,
) -> None:
    """Queue a speech event (resolves camera name from Protect API)."""
    from app.protect import get_protect_client
    from app.util import camera_display_name, find_camera

    try:
        client = await get_protect_client()
        camera = find_camera(client, camera_id)
        camera_name = camera_display_name(camera, camera_id)
    except Exception:
        camera_name = f"Unknown ({camera_id})"

    settings = get_settings()
    language = settings.get("language", "da")
    queue_transcription(event_id, camera_id, camera_name, timestamp_ms, language)
