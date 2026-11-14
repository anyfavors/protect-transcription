"""
Transcription CRUD routes: list, delete, retry, SRT download, bulk operations.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from app.auth import require_api_token
from app.config import AUDIO_PATH, LOCAL_TZ
from app.database import audit, get_connection
from app.util import (
    format_srt_time,
    parse_timestamp_to_ms,
    safe_audio_path,
    safe_download_filename,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/transcriptions")
async def get_transcriptions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=500),
    camera: str | None = None,
    date: str | None = None,
    search: str | None = None,
    status: str | None = None,
    cursor: str | None = Query(default=None, description="Keyset cursor: 'timestamp,id'"),
    date_from: str | None = None,
    date_to: str | None = None,
    starred: bool = False,
    visible: bool = Query(False, description="Hide errors, filtered and archived rows"),
):
    """
    List transcriptions.

    Two pagination modes:
    * Default OFFSET-based using ?page= and ?per_page= (compatible with older UI).
    * Keyset using ?cursor=<timestamp>,<id> for stable pagination on big tables.
      When ?cursor is given, ?page is ignored.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        rows, total = _query_transcriptions(
            cur,
            page,
            per_page,
            camera,
            date,
            search,
            status,
            cursor,
            date_from=date_from,
            date_to=date_to,
            starred=starred,
            visible=visible,
        )
        transcriptions = [_row_to_dict(r) for r in rows]

        next_cursor = None
        if rows and len(rows) == per_page:
            last = rows[-1]
            next_cursor = f"{last['timestamp']},{last['id']}"

        response: dict = {
            "transcriptions": transcriptions,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if total else 0,
        }
        if next_cursor:
            response["next_cursor"] = next_cursor
        return response

    finally:
        conn.close()


def _query_transcriptions(
    cur: sqlite3.Cursor,
    page: int,
    per_page: int,
    camera: str | None,
    date: str | None,
    search: str | None,
    status: str | None,
    cursor: str | None,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    starred: bool = False,
    visible: bool = False,
) -> tuple[list[sqlite3.Row], int]:
    where: list[str] = []
    params: list = []
    select = "t.*"

    if date_from:
        where.append("substr(t.timestamp, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        where.append("substr(t.timestamp, 1, 10) <= ?")
        params.append(date_to)
    if starred:
        where.append("t.starred = 1")
    if visible:
        where.append("t.status IN ('completed', 'pending', 'processing')")

    if camera:
        where.append("t.camera_name = ?")
        params.append(camera)
    if date:
        where.append("substr(t.timestamp, 1, 10) = ?")
        params.append(date)
    if status:
        where.append("t.status = ?")
        params.append(status)

    fts_query = fts_prefix_query(search) if search else ""
    if fts_query:
        where.insert(0, "transcriptions_fts MATCH ?")
        params.insert(0, fts_query)
        from_clause = (
            "transcriptions t INNER JOIN transcriptions_fts ON t.id = transcriptions_fts.rowid"
        )
        # \x02 / \x03 mark matches; the UI escapes the text, then turns them into <mark>.
        select = "t.*, snippet(transcriptions_fts, 0, char(2), char(3), '…', 24) AS snippet"
    else:
        from_clause = "transcriptions t"

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    cur.execute(f"SELECT COUNT(*) FROM {from_clause}{where_sql}", params)
    total = cur.fetchone()[0]

    keyset_clause = ""
    keyset_params: list = []
    if cursor:
        try:
            ts_str, id_str = cursor.rsplit(",", 1)
            cursor_id = int(id_str)
            keyset_clause = (
                " AND (t.timestamp < ? OR (t.timestamp = ? AND t.id < ?))"
                if where_sql
                else " WHERE (t.timestamp < ? OR (t.timestamp = ? AND t.id < ?))"
            )
            keyset_params = [ts_str, ts_str, cursor_id]
        except (ValueError, TypeError):
            pass

    if keyset_clause:
        sql = (
            f"SELECT {select} FROM {from_clause}{where_sql}{keyset_clause} "
            f"ORDER BY t.timestamp DESC, t.id DESC LIMIT ?"
        )
        cur.execute(sql, [*params, *keyset_params, per_page])
    else:
        offset = (page - 1) * per_page
        sql = (
            f"SELECT {select} FROM {from_clause}{where_sql} "
            f"ORDER BY t.timestamp DESC, t.id DESC LIMIT ? OFFSET ?"
        )
        cur.execute(sql, [*params, per_page, offset])

    return cur.fetchall(), total


def fts_prefix_query(search: str) -> str:
    """Turn free text into an FTS5 query: every word must match as a prefix.

    "hund skur" → '"hund"* AND "skur"*', so it also finds hunden, hundehus, skuret.
    """
    words = re.findall(r"\w+", search.lower())
    return " AND ".join(f'"{w}"*' for w in words)


@router.get("/api/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: int):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id=?", (transcription_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return _row_to_dict(row)


@router.patch("/api/transcriptions/{transcription_id}", dependencies=[Depends(require_api_token)])
async def update_transcription(transcription_id: int, request: Request):
    """Edit text (keeps the machine version), star/unstar, or set a note."""
    data = await request.json()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id=?", (transcription_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")
        fields: dict = {}
        if "starred" in data:
            fields["starred"] = 1 if data["starred"] else 0
        if "note" in data:
            note = str(data["note"] or "").strip()
            fields["note"] = note[:2000] or None
        if "transcription" in data:
            text = " ".join(str(data["transcription"] or "").split())
            if not text:
                raise HTTPException(status_code=400, detail="Text can't be empty")
            if row["edited_at"] is None:
                fields["original_transcription"] = row["transcription"]
            fields["transcription"] = text
            fields["status"] = "completed"
            fields["edited_at"] = datetime.now(tz=LOCAL_TZ).isoformat(timespec="seconds")
        if data.get("revert") and row["edited_at"] is not None:
            fields.update(
                transcription=row["original_transcription"],
                original_transcription=None,
                edited_at=None,
            )
        if not fields:
            raise HTTPException(status_code=400, detail="Nothing to update")
        cols = ", ".join(f"{k}=?" for k in fields)
        conn.execute(
            f"UPDATE transcriptions SET {cols} WHERE id=?", (*fields.values(), transcription_id)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id=?", (transcription_id,)
        ).fetchone()
    finally:
        conn.close()
    if "transcription" in fields:
        audit("edit_transcription", {"id": transcription_id})
    return _row_to_dict(row)


@router.post("/api/transcriptions/archive-errors", dependencies=[Depends(require_api_token)])
async def archive_errors():
    """Move errors that can never be fixed (no audio anywhere) out of the way."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE transcriptions SET status='archived' "
            "WHERE status='error' AND (audio_file IS NULL OR audio_file='')"
        )
        count = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    audit("archive_errors", {"count": count})
    return {"archived": count, "message": f"Archived {count} clips without audio"}


def _row_to_dict(row: sqlite3.Row) -> dict:
    segments = None
    try:
        raw = row["segments"]
        if raw:
            segments = json.loads(raw)
    except (KeyError, json.JSONDecodeError):
        pass

    keys = row.keys()
    return {
        "id": row["id"],
        "event_id": row["event_id"],
        "starred": bool(row["starred"]) if "starred" in keys else False,
        "note": row["note"] if "note" in keys else None,
        "edited_at": row["edited_at"] if "edited_at" in keys else None,
        "original_transcription": (
            row["original_transcription"] if "original_transcription" in keys else None
        ),
        "event_end": row["event_end"] if "event_end" in keys else None,
        "processed_at": row["processed_at"] if "processed_at" in keys else None,
        "snippet": row["snippet"] if "snippet" in keys else None,
        "camera_name": row["camera_name"],
        "timestamp": row["timestamp"],
        "transcription": row["transcription"],
        "segments": segments,
        "language": row["language"],
        "duration_seconds": row["duration_seconds"],
        "status": row["status"],
        "audio_file": row["audio_file"],
    }


@router.get("/api/audit-log", dependencies=[Depends(require_api_token)])
async def list_audit_log(limit: int = Query(100, ge=1, le=1000)):
    """Recent destructive actions, newest first."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, ts, action, details FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return {"entries": [dict(row) for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/cameras")
async def get_cameras():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT DISTINCT camera_name FROM transcriptions "
            "WHERE camera_name IS NOT NULL ORDER BY camera_name"
        )
        return {"cameras": [row[0] for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/dates")
async def get_dates():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT DISTINCT substr(timestamp, 1, 10) AS date FROM transcriptions "
            "WHERE timestamp IS NOT NULL ORDER BY date DESC LIMIT 90"
        )
        return {"dates": [row[0] for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/stats")
async def get_stats():
    """Status counters plus queue progress. 'today' is the local date."""
    from datetime import datetime as _dt

    today_local = _dt.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM transcriptions GROUP BY status")
        by_status = dict(cur.fetchall())
        cur.execute(
            "SELECT COUNT(*) FROM transcriptions WHERE substr(timestamp, 1, 10) = ?",
            (today_local,),
        )
        today = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM transcriptions "
            "WHERE processed_at >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-1 hour')"
        )
        rate_per_hour = cur.fetchone()[0]
    finally:
        conn.close()

    queued = by_status.get("pending", 0)
    return {
        "total": sum(by_status.values()),
        "completed": by_status.get("completed", 0),
        "processing": by_status.get("processing", 0),
        "queued": queued,
        "errors": by_status.get("error", 0),
        "filtered": by_status.get("filtered", 0),
        "archived": by_status.get("archived", 0),
        "today": today,
        "rate_per_hour": rate_per_hour,
        "eta_seconds": round(queued / rate_per_hour * 3600) if queued and rate_per_hour else None,
    }


@router.delete("/api/transcriptions/{transcription_id}", dependencies=[Depends(require_api_token)])
async def delete_transcription(transcription_id: int):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM transcriptions WHERE id=?", (transcription_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")

        if row["audio_file"]:
            safe = safe_audio_path(AUDIO_PATH, row["audio_file"])
            if safe is not None:
                try:
                    safe.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Could not delete audio file %s: %s", safe, exc)

        cursor.execute("DELETE FROM transcriptions WHERE id=?", (transcription_id,))
        conn.commit()
        audit("delete_transcription", {"id": transcription_id})
        return {"status": "deleted", "id": transcription_id}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error deleting transcription %d: %s", transcription_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/api/transcriptions/{transcription_id}/srt")
async def download_srt(transcription_id: int):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM transcriptions WHERE id=?", (transcription_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")

        segments = []
        if row["segments"]:
            with contextlib.suppress(json.JSONDecodeError):
                segments = json.loads(row["segments"])

        if not segments:
            segments = [
                {
                    "start": 0,
                    "end": row["duration_seconds"] or 10,
                    "text": row["transcription"] or "",
                }
            ]

        srt_lines = []
        for i, seg in enumerate(segments, 1):
            text = seg.get("text", "").strip()
            if text:
                srt_lines += [
                    str(i),
                    f"{format_srt_time(seg.get('start', 0))} --> "
                    f"{format_srt_time(seg.get('end', seg.get('start', 0) + 5))}",
                    text,
                    "",
                ]

        filename = safe_download_filename(
            f"{row['camera_name'] or 'unknown'}_{row['timestamp'] or 'unknown'}.srt"
        )
        return PlainTextResponse(
            content="\n".join(srt_lines),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        conn.close()


def _retry_one_in_txn(cur: sqlite3.Cursor, row: sqlite3.Row) -> tuple[str, str, str, int]:
    """Reset row to pending in-place. Returns (event_id, camera_id, camera_name, timestamp_ms)."""
    timestamp_ms = parse_timestamp_to_ms(str(row["timestamp"]))
    cur.execute(
        """
        UPDATE transcriptions
        SET status='pending', attempts=0, next_attempt_at=NULL, refetch=1, priority=1
        WHERE id=?
        """,
        (row["id"],),
    )
    return row["event_id"], row["camera_id"], row["camera_name"], timestamp_ms


@router.post(
    "/api/transcriptions/{transcription_id}/retry",
    dependencies=[Depends(require_api_token)],
)
async def retry_transcription(transcription_id: int):
    """Reset a single row to pending. Done in a single transaction (no data loss)."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM transcriptions WHERE id=?", (transcription_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Transcription not found")
        try:
            event_id, _, _, _ = _retry_one_in_txn(cur, row)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid timestamp: {row['timestamp']}"
            ) from exc
        conn.commit()
        return {"status": "queued", "id": transcription_id, "event_id": event_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error retrying transcription %d: %s", transcription_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/transcriptions/retry-errors", dependencies=[Depends(require_api_token)])
async def retry_all_errors():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM transcriptions WHERE status='error'")
        rows = cur.fetchall()
        if not rows:
            return {"queued": 0, "message": "No error transcriptions found"}

        queued = 0
        for row in rows:
            try:
                _retry_one_in_txn(cur, row)
                queued += 1
            except ValueError as exc:
                logger.error("Failed to parse timestamp for event %s: %s", row["event_id"], exc)
        conn.commit()
        return {"queued": queued, "message": f"Queued {queued} transcriptions for retry"}

    except Exception as exc:
        logger.exception("Error retrying all errors: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/transcriptions/retranscribe-all", dependencies=[Depends(require_api_token)])
async def retranscribe_all(request: Request):
    request_body = await request.json()
    include_errors: bool = request_body.get("include_errors", True)
    # Re-download audio from the NVR using the exact event window; rows whose
    # footage is gone fall back to the cached audio file.
    refetch: bool = request_body.get("refetch", True)

    statuses: tuple[str, ...] = (
        ("completed", "error", "filtered") if include_errors else ("completed",)
    )
    placeholders = ",".join("?" for _ in statuses)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE transcriptions
            SET status='pending', attempts=0, next_attempt_at=NULL, refetch=?
            WHERE status IN ({placeholders})
            """,
            (int(refetch), *statuses),
        )
        count = cursor.rowcount
        conn.commit()
    finally:
        conn.close()

    audit(
        "retranscribe_all", {"include_errors": include_errors, "refetch": refetch, "count": count}
    )
    return {"reset": count, "message": f"Queued {count} transcriptions for re-processing"}


@router.post("/api/transcriptions/bulk-delete", dependencies=[Depends(require_api_token)])
async def bulk_delete(request: Request):
    body = await request.json()
    ids: list[int] = body.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    if len(ids) > 1000:
        raise HTTPException(status_code=400, detail="Too many IDs (max 1000)")

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"SELECT id, audio_file FROM transcriptions WHERE id IN ({placeholders})",
            ids,
        )
        rows = cursor.fetchall()

        for row in rows:
            if row["audio_file"]:
                safe = safe_audio_path(AUDIO_PATH, row["audio_file"])
                if safe is not None:
                    try:
                        safe.unlink(missing_ok=True)
                    except Exception as exc:
                        logger.warning("Could not delete audio file %s: %s", safe, exc)

        cursor.execute(f"DELETE FROM transcriptions WHERE id IN ({placeholders})", ids)
        deleted = cursor.rowcount
        conn.commit()
        audit("bulk_delete", {"count": deleted, "ids": ids[:50]})
        return {"deleted": deleted, "message": f"Deleted {deleted} transcriptions"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in bulk delete: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/transcriptions/bulk-retry", dependencies=[Depends(require_api_token)])
async def bulk_retry(request: Request):
    body = await request.json()
    ids: list[int] = body.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    if len(ids) > 1000:
        raise HTTPException(status_code=400, detail="Too many IDs (max 1000)")

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in ids)
        cur.execute(f"SELECT * FROM transcriptions WHERE id IN ({placeholders})", ids)
        rows = cur.fetchall()

        queued = 0
        for row in rows:
            try:
                _retry_one_in_txn(cur, row)
                queued += 1
            except ValueError as exc:
                logger.error("Failed to parse timestamp for event %s: %s", row["event_id"], exc)
        conn.commit()
        return {"queued": queued, "message": f"Queued {queued} transcriptions for retry"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in bulk retry: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


_AUDIO_MIME = {".wav": "audio/wav", ".ogg": "audio/ogg", ".opus": "audio/opus"}


@router.get("/audio/{filename}", dependencies=[Depends(require_api_token)])
async def get_audio(filename: str):
    safe = safe_audio_path(AUDIO_PATH, filename)
    if safe is None:
        raise HTTPException(status_code=400, detail="invalid filename")
    if not safe.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    mime = _AUDIO_MIME.get(safe.suffix, "application/octet-stream")
    return FileResponse(safe, media_type=mime, filename=safe.name)


@router.post("/api/database/reset", dependencies=[Depends(require_api_token)])
async def reset_database(request: Request):
    """
    Drop all transcriptions, summaries, and audio files.
    Requires a confirmation body to prevent accidental fires.
    """
    body: dict = {}
    with contextlib.suppress(Exception):
        body = await request.json()
    if body.get("confirm") != "yes":
        raise HTTPException(
            status_code=400,
            detail='Provide {"confirm": "yes"} in body to confirm destructive reset',
        )

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM transcriptions")
        transcription_count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM transcriptions")
        cursor.execute("DELETE FROM summaries")
        conn.commit()
        cursor.execute("VACUUM")
    finally:
        conn.close()

    audio_files_deleted = 0
    audio_dir = Path(AUDIO_PATH)
    if audio_dir.exists():
        for f in audio_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix not in _AUDIO_MIME:
                continue
            try:
                f.unlink()
                audio_files_deleted += 1
            except Exception as exc:
                logger.warning("Failed to delete audio file %s: %s", f, exc)

    audit(
        "database_reset",
        {
            "transcriptions_deleted": transcription_count,
            "audio_files_deleted": audio_files_deleted,
        },
    )
    return {
        "status": "success",
        "message": "Database reset successfully",
        "transcriptions_deleted": transcription_count,
        "audio_files_deleted": audio_files_deleted,
    }
