"""Tests covering batch-2 fixes: filename sanitizer, query-token, audit log,
auto-sync auth, webhook body limit, empty-text handling, sync lock, /readyz alias."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime

import pytest

from app.util import safe_download_filename

# ── safe_download_filename ─────────────────────────────────────────────────


def test_safe_download_filename_strips_unsafe():
    out = safe_download_filename("front door <test>:bad?.srt")
    assert "<" not in out
    assert ">" not in out
    assert ":" not in out
    assert "?" not in out
    assert out.endswith(".srt")


def test_safe_download_filename_fallback():
    assert safe_download_filename("") == "file"
    assert safe_download_filename("///") == "file"


def test_safe_download_filename_preserves_normal():
    assert "Front_Door" in safe_download_filename("Front Door 2024-01-01.srt")


# ── /audio query-token auth ─────────────────────────────────────────────────


def test_audio_query_token_allows_browser_playback(client, tmp_db, monkeypatch, tmp_path):
    """Browsers can't send Authorization on <audio src=>, so ?token= must work."""
    from app import auth as auth_mod

    monkeypatch.setattr(auth_mod, "API_TOKEN", "topsecret")

    # Create a plausible audio file in the audio dir
    audio_dir = tmp_path / "audio"
    audio_file = audio_dir / "20240101_120000_Cam_aaaaaaaa.wav"
    audio_file.write_bytes(b"RIFFfake")

    r = client.get(f"/audio/{audio_file.name}")
    assert r.status_code == 401

    r = client.get(f"/audio/{audio_file.name}?token=topsecret")
    assert r.status_code == 200

    r = client.get(f"/audio/{audio_file.name}?token=wrong")
    assert r.status_code == 401


# ── /api/sync auth ─────────────────────────────────────────────────────────


def test_sync_requires_auth_when_token_set(client, monkeypatch):
    from app import auth as auth_mod

    monkeypatch.setattr(auth_mod, "API_TOKEN", "topsecret")
    r = client.post("/api/sync")
    assert r.status_code == 401


# ── Audit log ──────────────────────────────────────────────────────────────


def test_audit_log_records_destructive_actions(client, tmp_db):
    # Insert + delete a row to trigger audit
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO transcriptions (event_id, camera_id, camera_name, timestamp, status) "
        "VALUES ('e1', 'c1', 'Cam', ?, 'completed')",
        (datetime.now(tz=UTC).isoformat(),),
    )
    conn.commit()
    cur = conn.execute("SELECT id FROM transcriptions")
    row_id = cur.fetchone()[0]
    conn.close()

    client.delete(f"/api/transcriptions/{row_id}")

    r = client.get("/api/audit-log")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert any(e["action"] == "delete_transcription" for e in entries)


def test_database_reset_audited(client, tmp_db):
    client.post("/api/database/reset", json={"confirm": "yes"})
    r = client.get("/api/audit-log")
    actions = [e["action"] for e in r.json()["entries"]]
    assert "database_reset" in actions


# ── Webhook body size limit ────────────────────────────────────────────────


def test_webhook_rejects_oversized_body(client, monkeypatch):
    from app import auth as auth_mod

    monkeypatch.setattr(auth_mod, "WEBHOOK_SECRET", "")  # ensure size check still runs
    huge = b"x" * (2 * 1024 * 1024)  # 2 MiB > 1 MiB cap
    r = client.post("/api/webhook", content=huge)
    assert r.status_code == 413


# ── Empty text → filtered ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_returns_empty_flag(monkeypatch):
    from app import transcription as t

    async def fake_post(client, audio_data, model, data):
        return {"text": "   ", "language": "da"}

    monkeypatch.setattr(t, "_post_transcription", fake_post)
    monkeypatch.setattr(t, "get_settings", lambda: {"whisper_model": "m", "language": "da"})

    result = await t.transcribe_audio(b"fakeaudio")
    assert result.get("empty") is True


# ── Sync lock prevents concurrent runs ─────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_lock_serializes_calls(monkeypatch, tmp_db):
    from app import sync_service as ss

    call_count = 0
    in_flight = 0
    max_in_flight = 0

    async def fake_impl(hours: int) -> dict:
        nonlocal call_count, in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        call_count += 1
        return {"status": "completed", "events_queued": 0}

    monkeypatch.setattr(ss, "_run_sync_impl", fake_impl)

    await asyncio.gather(*(ss.run_sync(1) for _ in range(5)))
    assert call_count == 5
    assert max_in_flight == 1  # serialized


# ── /readyz alias ──────────────────────────────────────────────────────────


def test_readyz_alias(client):
    r = client.get("/readyz")
    # whisper unavailable in tests so readiness returns 503
    assert r.status_code in (200, 503)
    assert "checks" in r.json()


def test_livez_alias(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ── /metrics uptime gauge ──────────────────────────────────────────────────


def test_metrics_includes_uptime(client):
    r = client.get("/metrics")
    assert "process_uptime_seconds" in r.text


# ── Auto-summary catch-up ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollups_skip_when_no_transcripts(monkeypatch, tmp_db):
    from app import worker as wk

    # No transcripts in DB → rollups must not call generate
    called = []

    async def fake_gen(period: str, key: str) -> bool:
        called.append((period, key))
        return True

    monkeypatch.setattr(wk, "_generate_summary_safe", fake_gen)
    await wk.run_rollups()
    assert called == []
