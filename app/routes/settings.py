"""
Settings GET/PUT and connectivity test endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import require_api_token
from app.config import AVAILABLE_LANGUAGES, WHISPER_URL
from app.database import get_settings, save_setting
from app.http_clients import get_whisper_client
from app.protect import get_protect_client, get_protect_host, invalidate_protect_client

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_KEYS = {
    "whisper_model",
    "language",
    "buffer_before",
    "buffer_after",
    "vad_filter",
    "protect_host",
    "ollama_url",
    "ollama_model",
    "summary_language",
    "min_audio_energy",
    "audio_compression_days",
    "auto_sync_hours",
    "auto_sync_interval_minutes",
    "auto_summary_time",
    "processing_timeout_minutes",
    "protect_verify_ssl",
}

_BOOL_KEYS = {
    "vad_filter",
    "protect_verify_ssl",
}

_INT_BOUNDS = {
    "buffer_before": (1, 60),
    "buffer_after": (1, 600),
    "audio_compression_days": (0, 365),
    "auto_sync_hours": (1, 720),
    "auto_sync_interval_minutes": (5, 1440),
    "processing_timeout_minutes": (1, 1440),
}

# Settings that should never appear in the GET response (defence in depth —
# they aren't currently stored, but if a deployment ever adds them, redact).
_REDACTED_KEYS = {"protect_password", "api_token", "webhook_secret"}


@router.get("/api/settings")
async def api_get_settings():
    settings = {k: v for k, v in get_settings().items() if k not in _REDACTED_KEYS}
    return {
        "settings": settings,
        "available_languages": AVAILABLE_LANGUAGES,
    }


@router.put("/api/settings", dependencies=[Depends(require_api_token)])
async def api_update_settings(request: Request):
    data = await request.json()
    updated: list[str] = []
    protect_host_changed = False

    for key, value in data.items():
        if key not in _ALLOWED_KEYS:
            continue

        if key in _INT_BOUNDS:
            lo, hi = _INT_BOUNDS[key]
            try:
                int_val = int(value)
                if not (lo <= int_val <= hi):
                    raise ValueError(f"{key} must be between {lo} and {hi}")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if key in _BOOL_KEYS:
            value = "true" if value in (True, "true", "1", 1) else "false"

        if key == "protect_host":
            value = str(value).strip().rstrip("/").removeprefix("https://").removeprefix("http://")
            protect_host_changed = True

        if key == "protect_verify_ssl":
            protect_host_changed = True  # force reconnect to apply new SSL setting

        if key == "auto_summary_time":
            # Validate HH:MM
            try:
                h, m = str(value).split(":", 1)
                if not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
                    raise ValueError
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail="auto_summary_time must be HH:MM"
                ) from exc

        save_setting(key, str(value))
        updated.append(key)

    if protect_host_changed:
        invalidate_protect_client()
        logger.info("Protect host changed, client will reconnect on next request")

    settings = {k: v for k, v in get_settings().items() if k not in _REDACTED_KEYS}
    return {"status": "updated", "updated_keys": updated, "settings": settings}


@router.post("/api/settings/test-whisper")
async def test_whisper_connection():
    client = get_whisper_client()
    try:
        response = await client.get(f"{WHISPER_URL}/v1/models", timeout=10.0)
        if response.status_code == 200:
            return {
                "status": "connected",
                "whisper_url": WHISPER_URL,
                "models": response.json(),
            }
        return {"status": "error", "message": f"Whisper returned status {response.status_code}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@router.post("/api/settings/test-protect")
async def test_protect_connection():
    try:
        host = get_protect_host()
        if not host:
            return {"status": "error", "message": "Protect host not configured"}
        client = await get_protect_client(force_reconnect=True)
        nvr = client.bootstrap.nvr
        cameras = list(client.bootstrap.cameras.values())
        return {
            "status": "connected",
            "host": host,
            "nvr_name": nvr.name,
            "nvr_version": str(nvr.version),
            "camera_count": len(cameras),
            "cameras": [{"id": c.id, "name": c.name} for c in cameras],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@router.get("/api/settings/speaches-models")
async def get_speaches_models():
    """
    Returns all ASR models from the speaches registry merged with the list of
    already-installed models so the UI knows which ones need downloading.
    """
    client = get_whisper_client()
    installed_ids: set[str] = set()
    try:
        r = await client.get(f"{WHISPER_URL}/v1/models", timeout=15.0)
        if r.status_code == 200:
            payload = r.json()
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            for m in data:
                installed_ids.add(m.get("id", ""))
    except Exception as exc:
        logger.warning("Could not fetch installed models from speaches: %s", exc)

    registry: list[dict] = []
    try:
        r = await client.get(
            f"{WHISPER_URL}/v1/registry",
            params={"task": "automatic-speech-recognition"},
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json()
            registry = data.get("data", data) if isinstance(data, dict) else data
    except Exception as exc:
        logger.warning("Could not fetch speaches registry: %s", exc)

    models = [
        {
            "id": m.get("id", ""),
            "object": m.get("object", "model"),
            "owned_by": m.get("owned_by", ""),
            "language": m.get("language", []),
            "task": m.get("task", "automatic-speech-recognition"),
            "installed": m.get("id", "") in installed_ids,
        }
        for m in registry
    ]
    return {"models": models, "installed_ids": sorted(installed_ids)}


@router.post(
    "/api/settings/speaches-models/{model_id:path}",
    dependencies=[Depends(require_api_token)],
)
async def install_speaches_model(model_id: str):
    """Trigger speaches to download a model. May take several minutes."""
    if ".." in model_id or model_id.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid model id")

    logger.info("Requesting speaches to download model: %s", model_id)
    client = get_whisper_client()
    try:
        r = await client.post(f"{WHISPER_URL}/v1/models/{model_id}", timeout=600.0)
        if r.status_code in (200, 201):
            return {"status": "installed", "model_id": model_id}
        raise HTTPException(
            status_code=r.status_code,
            detail=f"speaches returned {r.status_code}: {r.text}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
