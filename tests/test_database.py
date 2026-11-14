"""Unit tests for app.database — schema init and settings CRUD."""

import sqlite3

from app.database import get_setting, get_settings, init_database, save_setting


def test_init_database_creates_tables(tmp_db):
    conn = sqlite3.connect(tmp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "transcriptions" in tables
    assert "settings" in tables
    assert "summaries" in tables


def test_init_database_creates_fts_index(tmp_db):
    conn = sqlite3.connect(tmp_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='transcriptions_fts'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_default_settings_populated(tmp_db):
    settings = get_settings()
    assert "whisper_model" in settings
    assert "language" in settings
    assert settings["language"] == "da"
    assert settings["vad_filter"] == "true"
    assert settings["summary_language"] == "English"
    assert "beam_size" not in settings


def test_save_and_retrieve_setting(tmp_db):
    save_setting("language", "en")
    assert get_setting("language") == "en"


def test_get_setting_returns_default_for_missing_key(tmp_db):
    result = get_setting("nonexistent_key", default="fallback")
    assert result == "fallback"


def test_get_setting_returns_none_when_no_default(tmp_db):
    assert get_setting("nonexistent_key") is None


def test_save_setting_overwrites_existing(tmp_db):
    save_setting("language", "sv")
    save_setting("language", "no")
    assert get_setting("language") == "no"


def test_init_database_is_idempotent(tmp_db):
    """Calling init_database() twice must not raise or duplicate rows."""
    init_database()
    settings = get_settings()
    # Default settings should not be duplicated
    assert isinstance(settings, dict)
    assert settings["language"] == "da"


def test_get_settings_returns_dict(tmp_db):
    result = get_settings()
    assert isinstance(result, dict)
    assert len(result) > 0
