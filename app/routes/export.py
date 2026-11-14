"""
Export endpoints: CSV, JSON, and bulk SRT (ZIP) downloads.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import sqlite3
import zipfile
from collections.abc import Iterator  # noqa: TC003 — used at runtime as generator return type
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import Response, StreamingResponse

from app.config import LOCAL_TZ
from app.database import get_connection
from app.util import csv_safe, format_srt_time, safe_download_filename

logger = logging.getLogger(__name__)
router = APIRouter()

# Hard cap on rows per export — defence against accidental OOM on huge tables.
_EXPORT_MAX_ROWS = 200_000


def _build_export_query(
    cursor: sqlite3.Cursor,
    camera: str | None,
    date_from: str | None,
    date_to: str | None,
    status: str | None,
    search: str | None,
    limit: int = _EXPORT_MAX_ROWS,
) -> Iterator[sqlite3.Row]:
    """Build and execute the export query with optional filters; yields rows lazily."""
    where = ["1=1"]
    params: list = []

    if camera:
        where.append("camera_name = ?")
        params.append(camera)
    if date_from:
        where.append("substr(timestamp, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        where.append("substr(timestamp, 1, 10) <= ?")
        params.append(date_to)
    if status:
        where.append("status = ?")
        params.append(status)
    if search:
        where.append(
            "id IN (SELECT rowid FROM transcriptions_fts WHERE transcriptions_fts MATCH ?)"
        )
        params.append(f'"{search.replace(chr(34), chr(34) + chr(34))}"')

    sql = (
        f"SELECT * FROM transcriptions WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?"
    )
    cursor.execute(sql, [*params, limit])
    while True:
        batch = cursor.fetchmany(500)
        if not batch:
            return
        yield from batch


def _generate_srt(row: sqlite3.Row) -> str:
    """Generate SRT content for a single transcription row."""
    segments = []
    if row["segments"]:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            segments = json.loads(row["segments"])

    if not segments:
        segments = [
            {
                "start": 0,
                "end": row["duration_seconds"] or 10,
                "text": row["transcription"] or "",
            }
        ]

    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker else ""
        start = format_srt_time(seg.get("start", 0))
        end = format_srt_time(seg.get("end", seg.get("start", 0) + 5))
        lines.extend([str(i), f"{start} --> {end}", f"{prefix}{text}", ""])

    return "\n".join(lines)


@router.get("/api/export/csv")
async def export_csv(
    camera: str | None = None,
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    status: str | None = None,
    search: str | None = None,
):
    """Export transcriptions as CSV (streamed; CSV-injection-safe)."""
    now = datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d_%H%M%S")

    def stream() -> Iterator[str]:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "id",
                    "event_id",
                    "camera_name",
                    "timestamp",
                    "status",
                    "language",
                    "confidence",
                    "duration_seconds",
                    "transcription",
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate()

            for row in _build_export_query(cursor, camera, date_from, date_to, status, search):
                writer.writerow(
                    [
                        row["id"],
                        csv_safe(row["event_id"]),
                        csv_safe(row["camera_name"]),
                        row["timestamp"],
                        row["status"],
                        row["language"],
                        row["confidence"],
                        row["duration_seconds"],
                        csv_safe(row["transcription"]),
                    ]
                )
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate()
        finally:
            conn.close()

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="transcriptions_{now}.csv"'},
    )


@router.get("/api/export/json")
async def export_json(
    camera: str | None = None,
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    status: str | None = None,
    search: str | None = None,
):
    """Export transcriptions as JSON (capped at _EXPORT_MAX_ROWS)."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        items: list[dict] = []
        for row in _build_export_query(cursor, camera, date_from, date_to, status, search):
            segments = None
            if row["segments"]:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    segments = json.loads(row["segments"])
            items.append(
                {
                    "id": row["id"],
                    "event_id": row["event_id"],
                    "camera_name": row["camera_name"],
                    "timestamp": row["timestamp"],
                    "status": row["status"],
                    "language": row["language"],
                    "confidence": row["confidence"],
                    "duration_seconds": row["duration_seconds"],
                    "transcription": row["transcription"],
                    "segments": segments,
                }
            )

        now = datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        return Response(
            content=json.dumps(items, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="transcriptions_{now}.json"'},
        )
    finally:
        conn.close()


@router.get("/api/export/srt")
async def export_srt_zip(
    camera: str | None = None,
    date_from: str | None = Query(None, alias="from"),
    date_to: str | None = Query(None, alias="to"),
    search: str | None = None,
):
    """Export all completed transcriptions as a ZIP of SRT files."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for row in _build_export_query(cursor, camera, date_from, date_to, "completed", search):
                srt = _generate_srt(row)
                if not srt.strip():
                    continue
                fname = safe_download_filename(
                    f"{row['camera_name'] or 'unknown'}_{row['timestamp'] or 'unknown'}.srt"
                )
                zf.writestr(fname, srt)

        now = datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="transcriptions_{now}.zip"'},
        )
    finally:
        conn.close()


# Backwards-compat alias for old SRT-route behaviour
_format_srt_time = format_srt_time
