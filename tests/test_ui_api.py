"""Endpoints behind the new UI: search, edits, stars, archive, jobs, ask, alerts, heatmap."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3

from tests.test_api_transcriptions import _insert_transcription

wk = importlib.import_module("app.worker")


def test_search_matches_prefixes_and_returns_snippet(client, tmp_db):
    _insert_transcription(tmp_db, event_id="a", transcription="hunden løb ind i hundehuset")
    _insert_transcription(tmp_db, event_id="b", transcription="katten sov")
    data = client.get("/api/transcriptions", params={"search": "hund"}).json()
    assert data["total"] == 1
    snippet = data["transcriptions"][0]["snippet"]
    assert "\x02hunden\x03" in snippet


def test_search_ignores_fts_syntax(client, tmp_db):
    _insert_transcription(tmp_db, event_id="a", transcription="hej med dig")
    # Quotes and operators are stripped to words; no FTS syntax error.
    assert client.get("/api/transcriptions", params={"search": 'NEAR( "'}).status_code == 200
    assert client.get("/api/transcriptions", params={"search": 'hej"*'}).json()["total"] == 1


def test_edit_keeps_original_and_rerun_preserves_edit(client, tmp_db):
    rid = _insert_transcription(tmp_db, transcription="maskin tekst")
    r = client.patch(f"/api/transcriptions/{rid}", json={"transcription": "rettet tekst"})
    assert r.status_code == 200
    body = r.json()
    assert (body["transcription"], body["original_transcription"]) == (
        "rettet tekst",
        "maskin tekst",
    )
    assert body["edited_at"]

    # A re-transcription must not overwrite the hand-corrected text.
    wk._finish(rid, "completed", "ny maskin tekst", "a.wav", 10.0, segments=None)
    row = client.get(f"/api/transcriptions/{rid}").json()
    assert row["transcription"] == "rettet tekst"
    assert row["original_transcription"] == "ny maskin tekst"

    reverted = client.patch(f"/api/transcriptions/{rid}", json={"revert": True}).json()
    assert reverted["transcription"] == "ny maskin tekst"
    assert reverted["edited_at"] is None


def test_star_note_and_filters(client, tmp_db):
    rid = _insert_transcription(tmp_db, event_id="s")
    _insert_transcription(tmp_db, event_id="e", status="error", transcription="Failed")
    client.patch(f"/api/transcriptions/{rid}", json={"starred": True, "note": " dyrlæge "})
    starred = client.get("/api/transcriptions", params={"starred": True}).json()
    assert [t["id"] for t in starred["transcriptions"]] == [rid]
    assert starred["transcriptions"][0]["note"] == "dyrlæge"
    visible = client.get("/api/transcriptions", params={"visible": True}).json()
    assert visible["total"] == 1


def test_archive_only_errors_without_audio(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", status="error")
    keep = _insert_transcription(tmp_db, event_id="e2", status="error")
    conn = sqlite3.connect(tmp_db)
    conn.execute("UPDATE transcriptions SET audio_file='x.wav' WHERE id=?", (keep,))
    conn.commit()
    conn.close()
    assert client.post("/api/transcriptions/archive-errors").json()["archived"] == 1
    stats = client.get("/api/stats").json()
    assert (stats["errors"], stats["archived"]) == (1, 1)


async def test_jobs_report_result_and_error():
    from app import jobs

    async def ok():
        return 42

    async def boom():
        raise RuntimeError("nope")

    a, b = jobs.start("t", ok()), jobs.start("t", boom())
    await asyncio.sleep(0.01)
    assert (jobs.get(a["id"])["status"], jobs.get(a["id"])["result"]) == ("done", 42)
    assert (jobs.get(b["id"])["status"], jobs.get(b["id"])["error"]) == ("error", "nope")


def test_alert_keywords_match_word_starts():
    from app.alerts import keywords, matches

    words = keywords({"alert_keywords": "Brand, hjælp ,"})
    assert words == ["brand", "hjælp"]
    assert matches("Der går en brandalarm!", words) == ["brand"]
    assert matches("ubrandet", words) == []


async def test_alert_sent_only_for_recent_events(tmp_db, monkeypatch):
    from datetime import datetime, timedelta

    from app import alerts
    from app.config import LOCAL_TZ
    from app.database import save_setting

    save_setting("alert_keywords", "hjælp")
    save_setting("alert_url", "http://ntfy.local/topic")
    sent = []

    async def fake_send(url, title, body):
        sent.append(title)
        return 200

    monkeypatch.setattr(alerts, "send", fake_send)
    now = datetime.now(tz=LOCAL_TZ)
    assert await alerts.check_alerts(1, "Indgang", now.isoformat(), "hjælp mig") == ["hjælp"]
    old = (now - timedelta(days=2)).isoformat()
    assert await alerts.check_alerts(2, "Indgang", old, "hjælp mig") == []
    assert len(sent) == 1 and "Indgang" in sent[0]


async def test_ask_uses_keywords_then_cites_clips(tmp_db, monkeypatch):
    from app import ask as ask_mod

    _insert_transcription(tmp_db, event_id="v", transcription="vi skal ringe til dyrlægen i morgen")
    replies = iter(["dyrlæge, ringe", "I går blev det aftalt at ringe til dyrlægen [#1]."])

    async def fake_ollama(system, user):
        return next(replies)

    monkeypatch.setattr(ask_mod, "_ollama", fake_ollama)
    out = await ask_mod.ask("When do we call the vet?")
    assert out["keywords"] == ["dyrlæge", "ringe"]
    assert [s["id"] for s in out["sources"]] == [1]
    assert "[#1]" in out["answer"]


def test_heatmap_uses_local_weekday_and_hour(client, tmp_db):
    # 2026-11-16 is a Monday; 00:30 local is Sunday 23:30 UTC.
    _insert_transcription(tmp_db, event_id="h", timestamp="2026-11-16T00:30:00+01:00")
    conn = sqlite3.connect(tmp_db)
    conn.execute("UPDATE transcriptions SET duration_seconds=30")
    conn.commit()
    conn.close()
    cells = client.get("/api/analytics/heatmap", params={"days": 3650}).json()["cells"]
    assert [(c["weekday"], c["hour"], c["count"]) for c in cells] == [(0, 0, 1)]


def test_me_reads_authentik_headers(client):
    r = client.get("/api/me", headers={"X-authentik-username": "ole", "X-authentik-name": "Ole"})
    assert r.json()["username"] == "ole"


def test_summary_generate_returns_job(client, tmp_db, monkeypatch):
    sm = importlib.import_module("app.summaries")

    async def fake_generate(period, key):
        return {"summary": "ok"}

    monkeypatch.setattr(sm, "generate_summary", fake_generate)
    r = client.post(
        "/api/summaries/generate", json={"period_type": "daily", "period_key": "2026-11-14"}
    )
    assert r.json()["status"] == "running"
    assert r.json()["kind"] == "summary"
