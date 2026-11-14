"""Hallucination rescue, event-window clips, deferral and automatic retries."""

from __future__ import annotations

import importlib
import sqlite3
from datetime import datetime, timedelta

import httpx

from app.config import LOCAL_TZ

tr = importlib.import_module("app.transcription")
wk = importlib.import_module("app.worker")


# ── text cleanup ────────────────────────────────────────────────────────────


def test_collapse_repeats_keeps_real_speech():
    assert tr._collapse_repeats("ja ja ja ja ja vi ses i morgen") == "ja vi ses i morgen"
    assert tr._collapse_repeats("nej nej nej") == "nej nej nej"  # under threshold
    loop = "det er jo en ferie " * 5
    assert tr._collapse_repeats(f"hej {loop}slut") == "hej det er jo en ferie slut"


def test_drop_bad_segments_rebuilds_text():
    result = {
        "text": "god tekst LOOP LOOP",
        "segments": [
            {"text": "god tekst", "compression_ratio": 1.2, "no_speech_prob": 0.1},
            {"text": "LOOP LOOP", "compression_ratio": 9.0, "no_speech_prob": 0.1},
            {"text": "støj", "compression_ratio": 1.0, "no_speech_prob": 0.9, "avg_logprob": -2},
        ],
    }
    out = tr._drop_bad_segments(result)
    assert out["text"] == "god tekst"
    assert len(out["segments"]) == 1


def test_text_rebuilt_with_spaces_between_segments():
    result = {
        "text": "mobiltelefonenDu var",
        "segments": [{"text": "mobiltelefonen"}, {"text": "Du var"}],
    }
    assert tr._drop_bad_segments(result)["text"] == "mobiltelefonen Du var"


# ── transcribe_audio against a fake speaches ────────────────────────────────


def _fake_whisper(monkeypatch, texts: list[str]) -> list[dict]:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("latin-1")
        calls.append({"prompt": 'name="prompt"' in body})
        return httpx.Response(200, json={"text": texts[len(calls) - 1], "segments": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(tr, "get_whisper_client", lambda: client)
    return calls


async def test_no_prompt_sent_and_loops_collapsed(tmp_db, monkeypatch):
    loop = "måske " * 12
    calls = _fake_whisper(monkeypatch, [f"vi tager til Sjælland i år {loop}"] * 2)
    out = await tr.transcribe_audio(b"RIFF")
    assert not any(c["prompt"] for c in calls)
    assert len(calls) == 2  # original + one retry
    assert out["text"] == "vi tager til Sjælland i år måske"


async def test_pure_loop_is_marked_hallucination(tmp_db, monkeypatch):
    _fake_whisper(monkeypatch, ["Ja. " * 20, "Ja. " * 20])
    out = await tr.transcribe_audio(b"RIFF")
    assert out.get("hallucination") is True
    assert "error" not in out


# ── worker: event windows, deferral, retries ────────────────────────────────


def _row(db: str, **extra) -> dict:
    conn = sqlite3.connect(db)
    cols = {"event_id": "6af8413a02922103e4200cbe", "camera_id": "cam", "camera_name": "Indgang"}
    cols.update(extra)
    cur = conn.execute(
        f"INSERT INTO transcriptions ({', '.join(cols)}, status) "
        f"VALUES ({', '.join('?' * len(cols))}, 'processing')",
        tuple(cols.values()),
    )
    conn.commit()
    row = dict(zip(("id", *cols), (cur.lastrowid, *cols.values()), strict=True))
    conn.close()
    return row


def _get(db: str, rid: int) -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM transcriptions WHERE id=?", (rid,)).fetchone()
    conn.close()
    return row


class _Event:
    def __init__(self, start: datetime, end: datetime | None):
        self.start, self.end = start, end


def _fake_protect(monkeypatch, event: _Event) -> None:
    class Client:
        async def get_event(self, _id):
            return event

    async def get_client():
        return Client()

    monkeypatch.setattr(wk, "get_protect_client", get_client)


async def test_ongoing_event_is_deferred(tmp_db, monkeypatch):
    now = datetime.now(tz=LOCAL_TZ)
    row = _row(tmp_db, timestamp=now.isoformat())
    _fake_protect(monkeypatch, _Event(now - timedelta(seconds=5), None))

    await wk.process_pending_transcription(row)

    stored = _get(tmp_db, row["id"])
    assert stored["status"] == "pending"
    assert stored["next_attempt_at"] is not None
    assert not wk._queue_busy()  # deferred rows don't count as ready


async def test_clip_spans_event_plus_buffers(tmp_db, monkeypatch):
    start = datetime(2026, 11, 14, 12, 0, 0, tzinfo=LOCAL_TZ)
    end = start + timedelta(seconds=40)
    row = _row(tmp_db, timestamp=(start + timedelta(seconds=14)).isoformat())
    _fake_protect(monkeypatch, _Event(start, end))
    windows = []

    async def fake_fetch(_cam, a, b):
        windows.append((a, b))
        return None  # NVR failure → retry path

    monkeypatch.setattr(wk, "fetch_audio_clip", fake_fetch)
    await wk.process_pending_transcription(row)

    assert windows == [(start - timedelta(seconds=5), end + timedelta(seconds=10))]
    stored = _get(tmp_db, row["id"])
    assert (stored["status"], stored["attempts"]) == ("pending", 1)  # backoff, not error


def test_retry_gives_up_after_max_attempts(tmp_db):
    row = _row(tmp_db, timestamp="2026-11-14T12:00:00+01:00")
    assert wk._retry_or_fail(row["id"], wk._MAX_ATTEMPTS - 1, "boom", "a.wav") == "error"
    stored = _get(tmp_db, row["id"])
    assert (stored["status"], stored["transcription"], stored["audio_file"]) == (
        "error",
        "boom",
        "a.wav",
    )


async def test_claim_prefers_new_events(tmp_db):
    old = _row(tmp_db, event_id="old", timestamp="2026-11-14T13:00:00+01:00", audio_file="x.wav")
    new = _row(tmp_db, event_id="new", timestamp="2026-11-14T12:00:00+01:00")
    conn = sqlite3.connect(tmp_db)
    conn.execute("UPDATE transcriptions SET status='pending'")
    conn.commit()
    conn.close()
    assert (await wk._claim_next_pending())["id"] == new["id"]
    assert (await wk._claim_next_pending())["id"] == old["id"]
