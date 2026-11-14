"""Summary rollups: local-day grouping, hierarchy, prompt content, scheduling."""

from __future__ import annotations

import importlib
import json
import sqlite3

import httpx
import pytest

sm = importlib.import_module("app.summaries")


def _insert(db: str, ts: str, text: str, camera: str = "Indgang") -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO transcriptions (event_id, camera_id, camera_name, timestamp, transcription,"
        " status) VALUES (?, 'c1', ?, ?, ?, 'completed')",
        (f"{ts}-{text}", camera, ts, text),
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def ollama(monkeypatch):
    """Fake Ollama /api/chat; records user prompts, replies 'SUMMARY n'."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["options"]["num_ctx"] == sm._NUM_CTX
        calls.append(body)
        return httpx.Response(200, json={"message": {"content": f"SUMMARY {len(calls)}"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(sm, "get_ollama_client", lambda: client)
    return calls


def test_daily_grouping_uses_local_date(tmp_db):
    # 00:30 local (+01:00) is 23:30 UTC the previous day; must stay on the local day.
    _insert(tmp_db, "2026-11-14T00:30:00.000000+01:00", "godmorgen alle sammen her")
    keys = [i["period_key"] for i in sm.get_summaries("daily")["items"]]
    assert keys == ["2026-11-14"]
    assert [i["period_key"] for i in sm.get_summaries("yearly")["items"]] == ["2026"]


async def test_daily_prompt_marks_cameras_as_locations_and_drops_fragments(tmp_db, ollama):
    _insert(tmp_db, "2026-11-14T10:00:00+01:00", "skal vi købe mælk i dag", "Forhaven")
    _insert(tmp_db, "2026-11-14T10:05:00+01:00", "Ja")
    await sm.generate_summary("daily", "2026-11-14")

    user = ollama[0]["messages"][1]["content"]
    assert "[10:00 @ Forhaven] skal vi købe mælk i dag" in user
    assert "Ja" not in user.split("\n\n")[1]
    system = ollama[0]["messages"][0]["content"]
    assert "NOT people" in system


async def test_monthly_rollup_builds_from_daily_summaries(tmp_db, ollama):
    _insert(tmp_db, "2026-10-01T10:00:00+02:00", "vi skal male hegnet i weekenden")
    _insert(tmp_db, "2026-10-02T10:00:00+02:00", "hunden skal til dyrlægen i morgen")

    out = await sm.generate_summary("monthly", "2026-10")

    # Two missing daily summaries generated first, then one monthly call over them.
    assert len(ollama) == 3
    monthly_prompt = ollama[2]["messages"][1]["content"]
    assert "=== 2026-10-01 ===\nSUMMARY" in monthly_prompt
    assert "male hegnet" not in monthly_prompt
    assert out["transcription_count"] == 2


async def test_large_day_is_chunked(tmp_db, ollama, monkeypatch):
    monkeypatch.setattr(sm, "_CHUNK_CHARS", 60)
    for i in range(4):
        _insert(tmp_db, f"2026-11-14T1{i}:00:00+01:00", f"en ret lang sætning nummer {i} her")
    await sm.generate_summary("daily", "2026-11-14")
    assert len(ollama) > 2  # several partial calls + one final merge
    assert "partial summaries" in ollama[-1]["messages"][1]["content"]


async def test_run_rollups_refreshes_current_and_fills_missing(tmp_db, monkeypatch):
    from app import worker as wk

    _insert(tmp_db, "2026-11-13T10:00:00+01:00", "noget der skete i går")
    monkeypatch.setattr(
        sm,
        "current_keys",
        lambda now=None: {
            "daily": "2026-11-14",
            "weekly": "2026-W45",
            "monthly": "2026-11",
            "yearly": "2026",
        },
    )
    called = []

    async def fake_gen(period: str, key: str) -> bool:
        called.append((period, key))
        return True

    monkeypatch.setattr(wk, "_generate_summary_safe", fake_gen)
    await wk.run_rollups(include_current=True)
    assert called == [
        ("daily", "2026-11-13"),
        ("weekly", "2026-W45"),
        ("monthly", "2026-11"),
        ("yearly", "2026"),
    ]
    called.clear()
    await wk.run_rollups(include_current=False)  # current periods wait for the nightly run
    assert called == [("daily", "2026-11-13")]


async def test_only_changed_content_is_stale_and_propagates(tmp_db, ollama):
    from app import worker as wk

    _insert(tmp_db, "2025-10-01T10:00:00+02:00", "vi skal male hegnet i weekenden")
    await sm.generate_summary("monthly", "2025-10")
    assert not any(i["stale"] for p in sm.PERIODS for i in sm.get_summaries(p)["items"])

    # Re-transcription changes the text of one row → its day, then month, go stale.
    conn = sqlite3.connect(tmp_db)
    conn.execute("UPDATE transcriptions SET transcription='vi skal male huset i weekenden'")
    conn.commit()
    conn.close()
    assert sm.get_summaries("daily")["items"][0]["stale"]

    before = len(ollama)
    await wk.run_rollups(include_current=False)
    # daily, then weekly + monthly (from the new daily), then yearly (from new monthly)
    assert len(ollama) - before == 4
    assert not any(i["stale"] for p in sm.PERIODS for i in sm.get_summaries(p)["items"])

    await wk.run_rollups(include_current=False)
    assert len(ollama) - before == 4  # nothing changed → nothing regenerated


def test_unknown_camera_names_backfilled(tmp_db):
    from app.database import init_database

    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO transcriptions (event_id, camera_id, camera_name, timestamp) "
        "VALUES ('a', 'MAC1', 'Indgang', '2026-01-01T00:00:00+01:00'),"
        "       ('b', 'MAC1', 'Unknown (MAC1)', '2026-01-02T00:00:00+01:00'),"
        "       ('c', 'MAC2', 'Unknown (MAC2)', '2026-01-02T00:00:00+01:00')"
    )
    conn.commit()
    init_database()
    names = dict(conn.execute("SELECT event_id, camera_name FROM transcriptions").fetchall())
    conn.close()
    assert names == {"a": "Indgang", "b": "Indgang", "c": "Unknown (MAC2)"}
