"""
Integration tests for the transcription REST endpoints.
All tests use the in-memory database fixture — no real NVR or Whisper needed.
"""

import sqlite3
from datetime import UTC, datetime


def _insert_transcription(db_path: str, **kwargs) -> int:
    """Helper: insert a transcription row and return its id."""
    defaults = {
        "event_id": "evt_001",
        "camera_id": "cam_abc",
        "camera_name": "Front Door",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "transcription": "Hej verden",
        "status": "completed",
        "language": "da",
    }
    defaults.update(kwargs)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transcriptions
            (event_id, camera_id, camera_name, timestamp, transcription, status, language)
        VALUES (:event_id, :camera_id, :camera_name, :timestamp, :transcription, :status, :language)
        """,
        defaults,
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ─────────────────────────────────────────────────────────────
# /api/stats
# ─────────────────────────────────────────────────────────────


def test_stats_empty(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["completed"] == 0
    assert data["errors"] == 0


def test_stats_counts(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", status="completed")
    _insert_transcription(tmp_db, event_id="e2", status="error")
    _insert_transcription(tmp_db, event_id="e3", status="processing")

    r = client.get("/api/stats")
    data = r.json()
    assert data["total"] == 3
    assert data["completed"] == 1
    assert data["errors"] == 1
    assert data["processing"] == 1


# ─────────────────────────────────────────────────────────────
# /api/transcriptions (list + filter)
# ─────────────────────────────────────────────────────────────


def test_list_transcriptions_empty(client):
    r = client.get("/api/transcriptions")
    assert r.status_code == 200
    data = r.json()
    assert data["transcriptions"] == []
    assert data["total"] == 0


def test_list_transcriptions_returns_row(client, tmp_db):
    _insert_transcription(tmp_db)
    r = client.get("/api/transcriptions")
    data = r.json()
    assert data["total"] == 1
    assert data["transcriptions"][0]["camera_name"] == "Front Door"


def test_filter_by_status(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", status="completed")
    _insert_transcription(tmp_db, event_id="e2", status="error")

    r = client.get("/api/transcriptions?status=error")
    data = r.json()
    assert data["total"] == 1
    assert data["transcriptions"][0]["status"] == "error"


def test_filter_by_camera(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", camera_name="Front Door")
    _insert_transcription(tmp_db, event_id="e2", camera_name="Back Yard")

    r = client.get("/api/transcriptions?camera=Back+Yard")
    data = r.json()
    assert data["total"] == 1
    assert data["transcriptions"][0]["camera_name"] == "Back Yard"


def test_full_text_search(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", transcription="Hej verden")
    _insert_transcription(tmp_db, event_id="e2", transcription="God morgen")

    r = client.get("/api/transcriptions?search=morgen")
    data = r.json()
    assert data["total"] == 1
    assert "morgen" in data["transcriptions"][0]["transcription"]


def test_pagination(client, tmp_db):
    for i in range(5):
        _insert_transcription(tmp_db, event_id=f"e{i}")

    r = client.get("/api/transcriptions?per_page=2&page=1")
    data = r.json()
    assert data["total"] == 5
    assert len(data["transcriptions"]) == 2
    assert data["pages"] == 3


# ─────────────────────────────────────────────────────────────
# /api/cameras and /api/dates
# ─────────────────────────────────────────────────────────────


def test_cameras_empty(client):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert r.json()["cameras"] == []


def test_cameras_distinct(client, tmp_db):
    _insert_transcription(tmp_db, event_id="e1", camera_name="Cam A")
    _insert_transcription(tmp_db, event_id="e2", camera_name="Cam A")
    _insert_transcription(tmp_db, event_id="e3", camera_name="Cam B")

    r = client.get("/api/cameras")
    cameras = r.json()["cameras"]
    assert sorted(cameras) == ["Cam A", "Cam B"]


def test_dates_empty(client):
    r = client.get("/api/dates")
    assert r.status_code == 200
    assert r.json()["dates"] == []


# ─────────────────────────────────────────────────────────────
# DELETE /api/transcriptions/{id}
# ─────────────────────────────────────────────────────────────


def test_delete_transcription(client, tmp_db):
    row_id = _insert_transcription(tmp_db)
    r = client.delete(f"/api/transcriptions/{row_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"

    r2 = client.get("/api/transcriptions")
    assert r2.json()["total"] == 0


def test_delete_nonexistent_returns_404(client):
    r = client.delete("/api/transcriptions/9999")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────
# POST /api/transcriptions/{id}/retry
# ─────────────────────────────────────────────────────────────


def test_retry_transcription_queues_pending(client, tmp_db):
    row_id = _insert_transcription(tmp_db, status="error", event_id="evt_retry")
    r = client.post(f"/api/transcriptions/{row_id}/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"

    # The old record was deleted and a new pending one inserted
    conn = sqlite3.connect(tmp_db)
    cur = conn.cursor()
    cur.execute("SELECT status FROM transcriptions WHERE event_id = 'evt_retry'")
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "pending"


def test_retry_nonexistent_returns_404(client):
    r = client.post("/api/transcriptions/9999/retry")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────
# POST /api/transcriptions/retry-errors
# ─────────────────────────────────────────────────────────────


def test_retry_all_errors(client, tmp_db):
    _insert_transcription(tmp_db, event_id="err1", status="error")
    _insert_transcription(tmp_db, event_id="err2", status="error")
    _insert_transcription(tmp_db, event_id="ok1", status="completed")

    r = client.post("/api/transcriptions/retry-errors")
    assert r.status_code == 200
    assert r.json()["queued"] == 2

    conn = sqlite3.connect(tmp_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transcriptions WHERE status='error'")
    assert cur.fetchone()[0] == 0
    conn.close()


def test_retry_all_errors_no_errors(client, tmp_db):
    _insert_transcription(tmp_db, event_id="ok1", status="completed")
    r = client.post("/api/transcriptions/retry-errors")
    assert r.status_code == 200
    assert r.json()["queued"] == 0


# ─────────────────────────────────────────────────────────────
# POST /api/transcriptions/retranscribe-all
# ─────────────────────────────────────────────────────────────


def test_retranscribe_all_completed_only(client, tmp_db):
    _insert_transcription(tmp_db, event_id="c1", status="completed", transcription="some text")
    _insert_transcription(tmp_db, event_id="e1", status="error", transcription="error text")

    r = client.post(
        "/api/transcriptions/retranscribe-all",
        json={"include_errors": False},
    )
    assert r.status_code == 200
    assert r.json()["reset"] == 1

    conn = sqlite3.connect(tmp_db)
    cur = conn.cursor()
    cur.execute("SELECT status, transcription, refetch FROM transcriptions WHERE event_id='c1'")
    row = cur.fetchone()
    conn.close()
    # Old text stays visible until the worker replaces it.
    assert row == ("pending", "some text", 1)


def test_retranscribe_all_include_errors(client, tmp_db):
    _insert_transcription(tmp_db, event_id="c1", status="completed")
    _insert_transcription(tmp_db, event_id="e1", status="error")

    r = client.post(
        "/api/transcriptions/retranscribe-all",
        json={"include_errors": True},
    )
    assert r.json()["reset"] == 2


# ─────────────────────────────────────────────────────────────
# GET /api/transcriptions/{id}/srt
# ─────────────────────────────────────────────────────────────


def test_srt_download(client, tmp_db):
    row_id = _insert_transcription(tmp_db, transcription="Hej verden")
    r = client.get(f"/api/transcriptions/{row_id}/srt")
    assert r.status_code == 200
    assert "Content-Disposition" in r.headers
    assert "Hej verden" in r.text


def test_srt_not_found(client):
    r = client.get("/api/transcriptions/9999/srt")
    assert r.status_code == 404
