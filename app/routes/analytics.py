"""
Analytics endpoints: hourly activity, daily trends, camera stats, language distribution.
"""

from fastapi import APIRouter, Query

from app.database import get_connection

router = APIRouter()


@router.get("/api/analytics/hourly")
async def hourly_activity(days: int = Query(30, ge=1, le=365)):
    """Speech events per hour of day (0-23)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT CAST(substr(timestamp, 12, 2) AS INTEGER) AS hour, COUNT(*) AS count
            FROM transcriptions
            WHERE status IN ('completed', 'filtered')
              AND timestamp >= datetime('now', 'localtime', ?)
            GROUP BY hour
            ORDER BY hour
            """,
            (f"-{days} days",),
        )
        data = {row[0]: row[1] for row in cursor.fetchall()}
        return {"hours": [{"hour": h, "count": data.get(h, 0)} for h in range(24)]}
    finally:
        conn.close()


@router.get("/api/analytics/daily")
async def daily_activity(days: int = Query(30, ge=1, le=365)):
    """Events per day for the last N days."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT substr(timestamp, 1, 10) AS day,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                   SUM(CASE WHEN status='filtered' THEN 1 ELSE 0 END) AS filtered
            FROM transcriptions
            WHERE timestamp >= datetime('now', 'localtime', ?)
            GROUP BY day
            ORDER BY day
            """,
            (f"-{days} days",),
        )
        cols = ["day", "total", "completed", "errors", "filtered"]
        return {"days": [dict(zip(cols, row, strict=False)) for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/analytics/cameras")
async def camera_stats():
    """Per-camera statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT camera_name,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                   SUM(CASE WHEN status='filtered' THEN 1 ELSE 0 END) AS filtered,
                   ROUND(AVG(CASE WHEN status='completed' THEN duration_seconds END), 1) AS avg_duration,
                   ROUND(AVG(CASE WHEN status='completed' THEN confidence END), 3) AS avg_confidence
            FROM transcriptions
            WHERE camera_name IS NOT NULL
            GROUP BY camera_name
            ORDER BY total DESC
        """)
        cols = [
            "camera_name",
            "total",
            "completed",
            "errors",
            "filtered",
            "avg_duration",
            "avg_confidence",
        ]
        return {"cameras": [dict(zip(cols, row, strict=False)) for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/analytics/languages")
async def language_stats():
    """Language distribution across completed transcriptions."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT COALESCE(language, 'unknown') AS language, COUNT(*) AS count
            FROM transcriptions
            WHERE status = 'completed' AND language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
        """)
        return {"languages": [{"language": row[0], "count": row[1]} for row in cursor.fetchall()]}
    finally:
        conn.close()


@router.get("/api/storage")
async def storage_stats():
    """Storage usage: audio files and database size."""
    from pathlib import Path

    from app.config import AUDIO_PATH, DATABASE_PATH

    audio_dir = Path(AUDIO_PATH)
    total_audio = 0
    wav_count = wav_size = ogg_count = ogg_size = 0

    if audio_dir.exists():
        for f in audio_dir.iterdir():
            if not f.is_file():
                continue
            size = f.stat().st_size
            total_audio += size
            if f.suffix == ".wav":
                wav_count += 1
                wav_size += size
            elif f.suffix == ".ogg":
                ogg_count += 1
                ogg_size += size

    db_path = Path(DATABASE_PATH)
    db_size = db_path.stat().st_size if db_path.exists() else 0
    # Include WAL and SHM files
    for suffix in ("-wal", "-shm"):
        p = db_path.with_name(db_path.name + suffix)
        if p.exists():
            db_size += p.stat().st_size

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE audio_file IS NOT NULL")
    files_in_db = cursor.fetchone()[0]
    conn.close()

    return {
        "total_bytes": total_audio + db_size,
        "audio_bytes": total_audio,
        "database_bytes": db_size,
        "wav_files": wav_count,
        "wav_bytes": wav_size,
        "ogg_files": ogg_count,
        "ogg_bytes": ogg_size,
        "files_referenced": files_in_db,
    }
