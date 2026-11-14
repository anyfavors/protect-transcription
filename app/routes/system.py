"""
UI support endpoints: who is signed in, service health, background jobs,
"ask the archive" and keyword-alert testing.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from app import jobs, protect
from app.alerts import keywords, send
from app.auth import require_api_token
from app.config import OLLAMA_URL, WHISPER_URL
from app.database import get_settings
from app.http_clients import get_ollama_client, get_whisper_client
from app.protect import get_protect_host

router = APIRouter()


@router.get("/api/me")
async def me(request: Request):
    """Identity forwarded by the Authentik proxy (empty when accessed directly)."""
    h = request.headers
    return {
        "username": h.get("x-authentik-username"),
        "name": h.get("x-authentik-name"),
        "email": h.get("x-authentik-email"),
    }


async def _check(coro) -> dict:
    start = time.monotonic()
    try:
        detail = await asyncio.wait_for(coro, timeout=8)
        return {"ok": True, "detail": detail, "ms": round((time.monotonic() - start) * 1000)}
    except TimeoutError:
        return {"ok": False, "detail": "No answer within 8 seconds"}
    except httpx.ConnectError as exc:
        return {"ok": False, "detail": f"Cannot connect: {exc}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc) or type(exc).__name__}


async def _protect() -> str:
    # Report the existing connection only: a health check must never trigger an
    # NVR login (an open browser tab polls this every minute). The worker reconnects.
    if not get_protect_host():
        raise RuntimeError("NVR address not set (System → Recording)")
    client = protect._protect_client
    if client is None:
        raise RuntimeError("Not connected yet; the app retries every 5 minutes")
    return f"{client.bootstrap.nvr.name} · {len(client.bootstrap.cameras)} cameras"


async def _whisper() -> str:
    r = await get_whisper_client().get(f"{WHISPER_URL}/v1/models", timeout=8)
    r.raise_for_status()
    model = get_settings().get("whisper_model", "")
    installed = [m["id"] for m in r.json().get("data", [])]
    return model if model in installed else f"{model} (not installed)"


async def _ollama() -> str:
    settings = get_settings()
    url = (settings.get("ollama_url") or OLLAMA_URL).rstrip("/")
    r = await get_ollama_client().get(f"{url}/api/tags", timeout=8)
    r.raise_for_status()
    model = settings.get("ollama_model", "")
    names = [m.get("name") for m in r.json().get("models", [])]
    return model if model in names else f"{model} (not pulled)"


@router.get("/api/system/health")
async def system_health():
    protect, whisper, ollama = await asyncio.gather(
        _check(_protect()), _check(_whisper()), _check(_ollama())
    )
    return {"protect": protect, "whisper": whisper, "ollama": ollama}


@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found (the server may have restarted)")
    return job


@router.post("/api/ask")
async def ask(request: Request):
    from app.ask import ask as run_ask

    question = str((await request.json()).get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    return jobs.start("ask", run_ask(question), question=question)


@router.post("/api/alerts/test", dependencies=[Depends(require_api_token)])
async def test_alert():
    settings = get_settings()
    url = settings.get("alert_url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Set an alert URL first")
    words = ", ".join(keywords(settings)) or "(no keywords set)"
    try:
        status = await send(url, "Test alert from Protect Transcribe", f"Watching for: {words}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach {url}: {exc}") from exc
    if status >= 400:
        raise HTTPException(status_code=502, detail=f"{url} answered HTTP {status}")
    return {"status": status}
