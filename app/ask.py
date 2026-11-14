"""
"Ask the archive": answer a question from the transcripts with the local LLM.

1. The LLM turns the question into Danish/English search words (the
   transcripts are Danish; questions may be English).
2. Full-text search picks the best-matching clips.
3. The LLM answers from those clips only, citing them as [#id].
"""

from __future__ import annotations

import logging
import re
import sqlite3

from fastapi import HTTPException

from app.database import get_connection
from app.summaries import _ollama

logger = logging.getLogger(__name__)

_MAX_SOURCES = 40
_MAX_CHARS_PER_SOURCE = 500


def _fts_query(words: list[str]) -> str:
    terms = []
    for w in words:
        tokens = re.findall(r"\w+", w.lower())
        if tokens:
            terms.append(" ".join(f'"{t}"*' for t in tokens))
    return " OR ".join(f"({t})" for t in terms)


def find_sources(words: list[str], limit: int = _MAX_SOURCES) -> list[dict]:
    query = _fts_query(words)
    if not query:
        return []
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT t.id, t.timestamp, t.camera_name, t.transcription
            FROM transcriptions_fts f JOIN transcriptions t ON t.id = f.rowid
            WHERE transcriptions_fts MATCH ? AND t.status = 'completed'
            ORDER BY bm25(transcriptions_fts) LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


async def ask(question: str) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    raw = await _ollama(
        "You turn questions about a Danish household's recorded conversations into "
        "full-text search words.",
        "Give 4-10 short search words (single words, no phrases) that would appear in "
        "Danish transcripts answering this question. Include Danish words, and English "
        "ones if the question is English. Use word stems (e.g. 'dyrlæge' not "
        "'dyrlægen'). Reply with the words only, comma-separated.\n\n"
        f"Question: {question}",
    )
    words = [w.strip() for w in re.split(r"[,\n]", raw) if 1 < len(w.strip()) < 40][:12]
    sources = find_sources(words)
    if not sources:
        return {"answer": None, "keywords": words, "sources": []}

    lines = []
    for s in sources:
        text = " ".join(str(s["transcription"]).split())[:_MAX_CHARS_PER_SOURCE]
        when = str(s["timestamp"])[:16].replace("T", " ")
        lines.append(f"[#{s['id']} | {when} | {s['camera_name']}] {text}")

    answer = await _ollama(
        "You answer questions about conversations recorded by microphones on cameras "
        "around a private home. Camera names (Indgang, Forhaven, ...) are locations, "
        "not people. Transcripts are automatic, mostly Danish, and contain errors. "
        "Answer only from the clips given. Cite every claim with the clip id in square "
        "brackets, like [#123]. If the clips don't answer the question, say so plainly. "
        "Answer in the language of the question, briefly.",
        f"Question: {question}\n\nClips:\n" + "\n".join(lines),
    )
    for s in sources:
        s["transcription"] = str(s["transcription"])[:300]
    return {"answer": answer, "keywords": words, "sources": sources}
