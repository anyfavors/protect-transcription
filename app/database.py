"""
SQLite database helpers: schema init, settings CRUD, and connection management.
All connections use WAL mode and a 30-second timeout to avoid lock contention.
"""

import contextlib
import logging
import sqlite3
from pathlib import Path

from app.config import DATABASE_PATH, OLLAMA_MODEL, OLLAMA_URL, PROTECT_HOST

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

_DEFAULT_SETTINGS: dict[str, str] = {
    "whisper_model": "Systran/faster-whisper-large-v3",
    "language": "da",
    # Clip = event start - buffer_before .. event END + buffer_after
    "buffer_before": "5",
    "buffer_after": "10",
    "vad_filter": "true",
    "protect_host": PROTECT_HOST,
    "ollama_url": OLLAMA_URL,
    "ollama_model": OLLAMA_MODEL,
    # Language the AI summaries are written in (free text, passed to the LLM)
    "summary_language": "English",
    # Noise / silence pre-filter: minimum RMS energy (0-1 normalised)
    "min_audio_energy": "0.005",
    # Audio compression: compress WAV to Opus/OGG after N days (0 = disabled)
    "audio_compression_days": "7",
    # Periodic background tasks
    "auto_sync_hours": "24",
    "auto_sync_interval_minutes": "60",
    "auto_summary_time": "23:55",
    # Stuck-row recovery: rows in 'processing' for longer than this are reclaimed
    "processing_timeout_minutes": "10",
    # Verify the Protect NVR's TLS certificate. Default false because home NVRs
    # ship with self-signed certs.
    "protect_verify_ssl": "false",
}


def get_connection() -> sqlite3.Connection:
    """Open and return a tuned SQLite connection to DATABASE_PATH."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Internal alias kept for backward compat within this module
_connect = get_connection


def _user_version(cur: sqlite3.Cursor) -> int:
    cur.execute("PRAGMA user_version")
    return int(cur.fetchone()[0])


def _set_user_version(cur: sqlite3.Cursor, version: int) -> None:
    cur.execute(f"PRAGMA user_version = {int(version)}")


def init_database() -> None:
    """Create tables, indexes, FTS virtual table, and triggers if they don't exist."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id         TEXT UNIQUE,
            camera_id        TEXT,
            camera_name      TEXT,
            timestamp        DATETIME,
            transcription    TEXT,
            segments         TEXT,
            language         TEXT,
            confidence       REAL,
            audio_file       TEXT,
            duration_seconds REAL,
            created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            status           TEXT DEFAULT 'pending'
        )
    """)

    for index_sql in (
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON transcriptions(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_camera    ON transcriptions(camera_name)",
        "CREATE INDEX IF NOT EXISTS idx_status    ON transcriptions(status)",
        "CREATE INDEX IF NOT EXISTS idx_status_ts ON transcriptions(status, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_ts_id     ON transcriptions(timestamp DESC, id DESC)",
    ):
        cur.execute(index_sql)

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
            transcription,
            camera_name,
            content='transcriptions',
            content_rowid='id'
        )
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ai AFTER INSERT ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(rowid, transcription, camera_name)
            VALUES (new.id, new.transcription, new.camera_name);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_ad AFTER DELETE ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, camera_name)
            VALUES ('delete', old.id, old.transcription, old.camera_name);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS transcriptions_au AFTER UPDATE ON transcriptions BEGIN
            INSERT INTO transcriptions_fts(transcriptions_fts, rowid, transcription, camera_name)
            VALUES ('delete', old.id, old.transcription, old.camera_name);
            INSERT INTO transcriptions_fts(rowid, transcription, camera_name)
            VALUES (new.id, new.transcription, new.camera_name);
        END
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    for key, value in _DEFAULT_SETTINGS.items():
        cur.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            period_type         TEXT NOT NULL,
            period_key          TEXT NOT NULL,
            period_label        TEXT,
            summary             TEXT,
            transcription_count INTEGER DEFAULT 0,
            generated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(period_type, period_key)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            action     TEXT NOT NULL,
            details    TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC)")

    # ── Migrations ──────────────────────────────────────────────────────────
    cur.execute("PRAGMA table_info(transcriptions)")
    columns = [col[1] for col in cur.fetchall()]
    for name, ddl in (
        ("segments", "TEXT"),
        # Resolved end of the Protect event (local ISO); clip = start-before .. end+after
        ("event_end", "TEXT"),
        # Automatic retry bookkeeping for transient failures / waiting for event end
        ("attempts", "INTEGER DEFAULT 0"),
        ("next_attempt_at", "TEXT"),  # UTC 'YYYY-MM-DDTHH:MM:SS'; NULL = ready now
        # 1 = re-download audio from the NVR (fall back to cached file if gone)
        ("refetch", "INTEGER DEFAULT 0"),
        # UTC 'YYYY-MM-DDTHH:MM:SS' of the last final result (rate / ETA, "re-transcribed")
        ("processed_at", "TEXT"),
    ):
        if name not in columns:
            logger.info("Migrating database: adding transcriptions.%s", name)
            cur.execute(f"ALTER TABLE transcriptions ADD COLUMN {name} {ddl}")

    cur.execute("PRAGMA table_info(summaries)")
    if "source_hash" not in [col[1] for col in cur.fetchall()]:
        logger.info("Migrating database: adding summaries.source_hash")
        cur.execute("ALTER TABLE summaries ADD COLUMN source_hash TEXT")

    # Rows queued while the Protect bootstrap was unavailable got "Unknown (<MAC>)"
    # names. Borrow the real name from another row with the same camera_id.
    cur.execute("""
        UPDATE transcriptions SET camera_name = (
            SELECT t.camera_name FROM transcriptions t
            WHERE t.camera_id = transcriptions.camera_id AND t.camera_name NOT LIKE 'Unknown (%'
            LIMIT 1
        )
        WHERE camera_name LIKE 'Unknown (%' AND EXISTS (
            SELECT 1 FROM transcriptions t
            WHERE t.camera_id = transcriptions.camera_id AND t.camera_name NOT LIKE 'Unknown (%'
        )
    """)
    if cur.rowcount:
        logger.info("Backfilled camera names on %d rows", cur.rowcount)

    current_version = _user_version(cur)
    if current_version < SCHEMA_VERSION:
        logger.info("Migrating schema: %d → %d", current_version, SCHEMA_VERSION)
        # FTS rebuild only when version bumps (or first run)
        with contextlib.suppress(sqlite3.OperationalError):
            cur.execute("INSERT INTO transcriptions_fts(transcriptions_fts) VALUES('rebuild')")
        if 0 < current_version < 3:
            # buffer_after used to be measured from the event START (so it had to
            # be large); it is now measured from the event END.
            cur.execute(
                "UPDATE settings SET value='10' WHERE key='buffer_after' AND CAST(value AS INT) > 60"
            )
            if cur.rowcount:
                logger.info("buffer_after now counts from event end; reset to 10s")
        _set_user_version(cur, SCHEMA_VERSION)

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s with FTS5 support", DATABASE_PATH)


def get_settings() -> dict[str, str]:
    """Return all settings as a plain dict."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def get_setting(key: str, default: str | None = None) -> str | None:
    """Return a single setting value, or *default* if not found."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else default
    finally:
        conn.close()


def save_setting(key: str, value: str) -> None:
    """Upsert a single setting."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("Setting saved: %s = %s", key, value)


def audit(action: str, details: str | dict | None = None) -> None:
    """
    Append an entry to the audit_log table.

    Failures are swallowed (audit logging must never break the actual operation).
    """
    import json as _json

    try:
        if isinstance(details, dict):
            details_str = _json.dumps(details, ensure_ascii=False)[:2000]
        elif details is None:
            details_str = None
        else:
            details_str = str(details)[:2000]

        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO audit_log (action, details) VALUES (?, ?)",
                (action, details_str),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - audit must never raise
        logger.warning("Audit log write failed for %s: %s", action, exc)


def reset_stuck_processing(timeout_minutes: int) -> int:
    """
    Move rows stuck in 'processing' for longer than *timeout_minutes* back to 'pending'.

    Returns the number of rows reset. Used at startup and periodically by the worker.
    """
    if timeout_minutes <= 0:
        return 0
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE transcriptions
            SET status='pending'
            WHERE status='processing'
              AND datetime(COALESCE(created_at, timestamp))
                  <= datetime('now', ?)
            """,
            (f"-{timeout_minutes} minutes",),
        )
        count = cur.rowcount
        conn.commit()
        if count:
            logger.info("Reset %d stuck 'processing' rows to 'pending'", count)
        return count
    finally:
        conn.close()
