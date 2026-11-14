"""
AI summary generation via Ollama.

Rollups are hierarchical so every LLM call stays inside the context window:
  daily   ← raw transcriptions (chunked map-reduce when a day is large)
  weekly  ← daily summaries
  monthly ← daily summaries
  yearly  ← monthly summaries
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime

from fastapi import HTTPException

from app.config import LOCAL_TZ, OLLAMA_MODEL, OLLAMA_URL
from app.database import get_connection, get_settings
from app.http_clients import get_ollama_client

logger = logging.getLogger(__name__)

# Timestamps are stored as local-time ISO strings with an offset
# ("2026-11-14T00:30:00+01:00"). SQLite date functions would convert those
# to UTC and shift late-night events into the previous day, so group on the
# local wall-clock prefix instead.
_PERIOD_EXPR = {
    "daily": "substr(timestamp, 1, 10)",
    "weekly": "strftime('%Y-W%W', substr(timestamp, 1, 10))",
    "monthly": "substr(timestamp, 1, 7)",
    "yearly": "substr(timestamp, 1, 4)",
}
PERIODS = tuple(_PERIOD_EXPR)

# parent period → (child period, SQL expr mapping a child period_key to the parent key)
_ROLLUP_SOURCE = {
    "weekly": ("daily", "strftime('%Y-W%W', period_key)"),
    "monthly": ("daily", "substr(period_key, 1, 7)"),
    "yearly": ("monthly", "substr(period_key, 1, 4)"),
}

# Ollama context window for summary calls; input chunks are sized to fit.
# ponytail: char-based budget (~3.5 chars/token), use a tokenizer if chunks overflow.
_NUM_CTX = 12288
_CHUNK_CHARS = 24000

# Transcripts with fewer words than this ("Ja", "!", "3") carry no content.
_MIN_WORDS = 3

_COMPLETED = "status = 'completed' AND transcription IS NOT NULL AND transcription != ''"


def _period_expr(period: str) -> str:
    if period not in _PERIOD_EXPR:
        raise ValueError(f"period must be one of {list(_PERIOD_EXPR)}")
    return _PERIOD_EXPR[period]


def _count(cursor: sqlite3.Cursor, period: str, period_key: str) -> int:
    cursor.execute(
        f"SELECT COUNT(*) FROM transcriptions WHERE {_COMPLETED} AND {_period_expr(period)} = ?",
        (period_key,),
    )
    return int(cursor.fetchone()[0])


def _source_hashes(cursor: sqlite3.Cursor, period: str, key: str | None = None) -> dict[str, str]:
    """
    Fingerprint of what each summary is built from, keyed by period_key.

    daily: the period's transcription texts; rollups: their child summaries.
    A summary is stale when its stored source_hash differs, so only real
    content changes trigger regeneration (and propagate upwards level by level).
    """
    if period == "daily":
        sql = (
            f"SELECT {_PERIOD_EXPR['daily']} AS k, id, transcription FROM transcriptions "
            f"WHERE {_COMPLETED}"
        )
        args: tuple = ()
        if key is not None:
            sql += f" AND {_PERIOD_EXPR['daily']} = ?"
            args = (key,)
        cursor.execute(sql + " ORDER BY k, id", args)
    else:
        child, expr = _ROLLUP_SOURCE[period]
        sql = (
            f"SELECT {expr} AS k, period_key, summary FROM summaries "
            "WHERE period_type = ? AND summary IS NOT NULL"
        )
        args = (child,)
        if key is not None:
            sql += f" AND {expr} = ?"
            args = (child, key)
        cursor.execute(sql + " ORDER BY k, period_key", args)

    hashes: dict[str, hashlib._Hash] = {}
    for k, ident, text in cursor.fetchall():
        h = hashes.setdefault(k, hashlib.sha1())
        h.update(f"{ident}\x1f{text}\x1e".encode())
    return {k: h.hexdigest() for k, h in hashes.items()}


def get_summaries(period: str, limit: int | None = 60) -> dict:
    """Return stored summaries and periods with transcriptions for the given period type."""
    expr = _period_expr(period)

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {expr} AS period_key, COUNT(*) AS cnt
            FROM transcriptions
            WHERE {_COMPLETED}
            GROUP BY period_key
            ORDER BY period_key DESC
            LIMIT ?
            """,
            (-1 if limit is None else limit,),
        )
        periods = [
            {"period_key": row["period_key"], "count": row["cnt"]} for row in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT period_key, summary, source_hash, generated_at "
            "FROM summaries WHERE period_type = ?",
            (period,),
        )
        summaries_map = {row["period_key"]: dict(row) for row in cursor.fetchall()}
        hashes = _source_hashes(cursor, period)
    finally:
        conn.close()

    result = []
    for p in periods:
        key = p["period_key"]
        s = summaries_map.get(key)
        result.append(
            {
                "period_key": key,
                "count": p["count"],
                "summary": s["summary"] if s else None,
                "generated_at": s["generated_at"] if s else None,
                "stale": s is not None and s["source_hash"] != hashes.get(key),
            }
        )

    return {"period": period, "items": result}


def backfill_source_hashes() -> int:
    """
    One-off for summaries created before source_hash existed: adopt the current
    hash where the transcription count still matches, so unchanged periods are
    not all regenerated at once. Mismatched ones stay NULL (= stale).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        filled = 0
        for period in PERIODS:
            cursor.execute(
                "SELECT period_key, transcription_count FROM summaries "
                "WHERE period_type = ? AND source_hash IS NULL",
                (period,),
            )
            rows = cursor.fetchall()
            if not rows:
                continue
            hashes = _source_hashes(cursor, period)
            for key, stored_count in rows:
                if key in hashes and _count(cursor, period, key) == stored_count:
                    cursor.execute(
                        "UPDATE summaries SET source_hash = ? "
                        "WHERE period_type = ? AND period_key = ?",
                        (hashes[key], period, key),
                    )
                    filled += 1
        conn.commit()
        if filled:
            logger.info("Backfilled source_hash on %d existing summaries", filled)
        return filled
    finally:
        conn.close()


def _system_prompt(language: str) -> str:
    return (
        "You summarise audio picked up by microphones on security cameras around a "
        "private home. Each line is tagged with the time and the camera LOCATION where "
        "the audio was recorded (e.g. Indgang = front entrance, Forhaven = front garden, "
        "Baghaven = back garden, Drivhus = greenhouse). "
        "Locations are places, NOT people: never write that a location said, discussed "
        "or did something. The speakers are unknown — refer to them as 'someone', "
        "'people' or 'the household', and use the location only to say where something "
        "was overheard. "
        "Transcripts are automatic and mostly Danish (some English); they contain errors "
        "and fragments, so ignore fragments that make no sense and do not invent details. "
        f"Write the summary in {language}. Be concise and factual."
    )


async def _ollama(system: str, user: str) -> str:
    settings = get_settings()
    ollama_url = (settings.get("ollama_url") or OLLAMA_URL).rstrip("/")
    ollama_model = settings.get("ollama_model") or OLLAMA_MODEL

    client = get_ollama_client()
    try:
        # Native endpoint: unlike /v1/chat/completions it lets us set num_ctx.
        # Ollama's default context silently truncates long prompts.
        response = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"num_ctx": _NUM_CTX},
            },
            timeout=300.0,
        )
    except Exception as exc:
        logger.exception("Error calling Ollama: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to reach Ollama: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {response.status_code}: {response.text[:300]}",
        )
    try:
        return str(response.json()["message"]["content"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response: {exc}") from exc


def _chunks(lines: list[str], limit: int) -> list[list[str]]:
    out: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if out[-1] and size + len(line) > limit:
            out.append([])
            size = 0
        out[-1].append(line[:limit])
        size += len(line) + 1
    return out


async def _summarise(lines: list[str], label: str, what: str, instructions: str) -> str:
    """Summarise *lines*, map-reducing over chunks when they exceed the context budget."""
    language = get_settings().get("summary_language") or "English"
    system = _system_prompt(language)

    parts = _chunks(lines, _CHUNK_CHARS)
    if len(parts) > 1:
        logger.info("Summary for %s: %d lines split into %d chunks", label, len(lines), len(parts))
        partials = []
        for i, part in enumerate(parts, 1):
            body = "\n".join(part)
            partials.append(
                await _ollama(
                    system,
                    f"Part {i} of {len(parts)} of the {what} from {label}:\n\n{body}\n\n"
                    "List the notable topics and events in short bullet points with times.",
                )
            )
        lines = [f"Part {i}:\n{p}" for i, p in enumerate(partials, 1)]
        what = "partial summaries"

    body = "\n".join(lines)
    return await _ollama(system, f"Here are the {what} from {label}:\n\n{body}\n\n{instructions}")


def _transcript_lines(cursor: sqlite3.Cursor, period_key: str) -> list[str]:
    cursor.execute(
        f"""
        SELECT camera_name, timestamp, transcription
        FROM transcriptions
        WHERE {_COMPLETED} AND {_PERIOD_EXPR["daily"]} = ?
        ORDER BY timestamp ASC
        """,
        (period_key,),
    )
    lines = []
    for row in cursor.fetchall():
        text = " ".join(str(row["transcription"]).split())
        if len(text.split()) < _MIN_WORDS:
            continue
        ts = str(row["timestamp"])[11:16]
        lines.append(f"[{ts} @ {row['camera_name']}] {text}")
    return lines


def _child_summaries(cursor: sqlite3.Cursor, period: str, period_key: str) -> list[str]:
    child, expr = _ROLLUP_SOURCE[period]
    cursor.execute(
        f"SELECT period_key, summary FROM summaries "
        f"WHERE period_type = ? AND {expr} = ? AND summary IS NOT NULL ORDER BY period_key",
        (child, period_key),
    )
    return [f"=== {r['period_key']} ===\n{r['summary']}" for r in cursor.fetchall()]


def _missing_children(cursor: sqlite3.Cursor, period: str, period_key: str) -> list[str]:
    """Child period keys that have transcriptions but no stored summary yet."""
    child, expr = _ROLLUP_SOURCE[period]
    child_expr = _PERIOD_EXPR[child]
    cursor.execute(
        f"""
        SELECT DISTINCT {child_expr} AS k FROM transcriptions
        WHERE {_COMPLETED} AND {_PERIOD_EXPR[period]} = ?
          AND {child_expr} NOT IN (SELECT period_key FROM summaries WHERE period_type = ?)
        ORDER BY k
        """,
        (period_key, child),
    )
    return [r[0] for r in cursor.fetchall()]


def _label(period: str, period_key: str) -> str:
    if period == "daily":
        try:
            return datetime.strptime(period_key, "%Y-%m-%d").strftime("%A %Y-%m-%d")
        except ValueError:
            return period_key
    return {"weekly": "week ", "monthly": "month ", "yearly": "year "}[period] + period_key


async def generate_summary(period: str, period_key: str) -> dict:
    """
    Generate (or regenerate) an AI summary for the given period via Ollama.
    Rollups first generate any missing child summaries they depend on.
    Raises HTTPException on bad input or upstream failure.
    """
    _period_expr(period)  # validates

    if period in _ROLLUP_SOURCE:
        conn = get_connection()
        try:
            missing = _missing_children(conn.cursor(), period, period_key)
        finally:
            conn.close()
        child = _ROLLUP_SOURCE[period][0]
        for key in missing:
            logger.info("Generating missing %s summary %s for %s %s", child, key, period, key)
            await generate_summary(child, key)

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        count = _count(cursor, period, period_key)
        if not count:
            raise HTTPException(status_code=404, detail="No transcriptions found for this period")

        label = _label(period, period_key)
        # Hash the inputs before the (slow) LLM call: changes arriving meanwhile
        # leave the summary stale so the next pass picks them up.
        source_hash = _source_hashes(cursor, period, period_key).get(period_key)
        if period == "daily":
            lines = _transcript_lines(cursor, period_key)
            if not lines:
                summary_text = "No meaningful speech was captured."
            else:
                summary_text = await _summarise(
                    lines,
                    label,
                    "transcriptions",
                    "Summarise what was talked about, grouped by theme or time of day. "
                    "Keep it under 200 words.",
                )
        else:
            lines = _child_summaries(cursor, period, period_key)
            summary_text = await _summarise(
                lines,
                label,
                f"{_ROLLUP_SOURCE[period][0]} summaries",
                f"Write an overview of the {period.removesuffix('ly')} "
                "highlighting recurring themes, notable events and changes over time. "
                "Do not repeat every detail. Keep it under 300 words.",
            )

        cursor.execute(
            """
            INSERT INTO summaries
                (period_type, period_key, period_label, summary, transcription_count,
                 source_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(period_type, period_key) DO UPDATE SET
                summary             = excluded.summary,
                transcription_count = excluded.transcription_count,
                period_label        = excluded.period_label,
                source_hash         = excluded.source_hash,
                generated_at        = CURRENT_TIMESTAMP
            """,
            (period, period_key, label, summary_text, count, source_hash),
        )
        conn.commit()

        logger.info("Generated %s summary for %s (%d transcriptions)", period, period_key, count)
        return {
            "period_type": period,
            "period_key": period_key,
            "summary": summary_text,
            "transcription_count": count,
        }
    finally:
        conn.close()


def current_keys(now: datetime | None = None) -> dict[str, str]:
    """Period keys containing *now* (local time), matching _PERIOD_EXPR formats."""
    now = now or datetime.now(tz=LOCAL_TZ)
    return {
        "daily": now.strftime("%Y-%m-%d"),
        "weekly": now.strftime("%Y-W%W"),
        "monthly": now.strftime("%Y-%m"),
        "yearly": now.strftime("%Y"),
    }
