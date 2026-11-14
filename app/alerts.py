"""
Keyword alerts: notify an HTTP endpoint when chosen words are heard.

Settings: ``alert_keywords`` (comma-separated, matched at word starts, so
"brand" also matches "brandalarm") and ``alert_url``. The message is POSTed as
plain text with a ``Title`` header, which works for ntfy topics and Home
Assistant webhooks alike.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import httpx

from app.config import LOCAL_TZ
from app.database import audit, get_settings
from app.metrics import inc

logger = logging.getLogger(__name__)

# Only alert on events this recent (never on backfills or re-runs of old clips).
_MAX_AGE = timedelta(hours=6)


def keywords(settings: dict[str, str]) -> list[str]:
    return [w.strip().lower() for w in settings.get("alert_keywords", "").split(",") if w.strip()]


def matches(text: str, words: list[str]) -> list[str]:
    lowered = text.lower()
    return [w for w in words if re.search(rf"(?<!\w){re.escape(w)}", lowered)]


async def send(url: str, title: str, body: str) -> int:
    # Header values must be latin-1; Danish letters are, anything else is replaced.
    safe_title = title.encode("latin-1", "replace").decode("latin-1")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            url, content=body.encode(), headers={"Title": safe_title, "Tags": "speech_balloon"}
        )
    return resp.status_code


async def check_alerts(record_id: int, camera_name: str, timestamp: str, text: str) -> list[str]:
    """Send an alert if *text* contains a keyword. Never raises."""
    try:
        settings = get_settings()
        words = keywords(settings)
        url = settings.get("alert_url", "").strip()
        if not words or not url or not text:
            return []
        when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=LOCAL_TZ)
        if datetime.now(tz=LOCAL_TZ) - when > _MAX_AGE:
            return []
        hits = matches(text, words)
        if not hits:
            return []
        title = f'"{hits[0]}" heard at {camera_name} {when.astimezone(LOCAL_TZ):%H:%M}'
        status = await send(url, title, text[:1000])
        audit(
            "keyword_alert",
            {"id": record_id, "keywords": hits, "camera": camera_name, "http_status": status},
        )
        inc("alerts_sent_total")
        logger.info("Keyword alert %s for #%d (HTTP %d)", hits, record_id, status)
        return hits
    except Exception as exc:
        logger.warning("Keyword alert failed for #%d: %s", record_id, exc)
        return []
