"""
Protect Transcription Service
==============================
FastAPI application factory.  Import ``app`` to get the ASGI app instance.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.auth import auth_enabled, webhook_auth_enabled
from app.config import AUDIO_PATH
from app.database import init_database, reset_stuck_processing
from app.http_clients import close_clients
from app.logging_config import configure_logging
from app.middleware import RequestIdMiddleware, install_log_filter
from app.protect import close_protect_client, get_protect_client
from app.routes import (
    analytics,
    export,
    health,
    settings,
    summaries,
    sync,
    system,
    transcriptions,
    webhook,
    ws,
)
from app.worker import (
    audio_compression_worker,
    auto_summary_worker,
    auto_sync_worker,
    protect_health_worker,
    stuck_processing_recovery_worker,
    transcription_worker,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _supervised(name: str, factory):
    """
    Wrap a worker coroutine factory in a supervisor that restarts it
    if it exits unexpectedly. Cancellation is propagated cleanly.
    """

    async def _runner() -> None:
        while True:
            try:
                await factory()
                logger.warning("Worker %s exited cleanly; restarting in 5s", name)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Worker %s crashed: %s — restarting in 10s", name, exc)
                await asyncio.sleep(10)

    return _runner


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup
    configure_logging()
    install_log_filter()
    init_database()
    Path(AUDIO_PATH).mkdir(parents=True, exist_ok=True)

    # Reclaim rows that were processing when the previous pod was killed.
    reset_stuck_processing(timeout_minutes=10)

    if not auth_enabled():
        logger.warning(
            "API_TOKEN env var not set — destructive endpoints are UNAUTHENTICATED. "
            "Set API_TOKEN to enable bearer-token auth."
        )
    if not webhook_auth_enabled():
        logger.warning(
            "WEBHOOK_SECRET env var not set — /api/webhook accepts unsigned requests. "
            "Set WEBHOOK_SECRET to enable HMAC verification."
        )

    try:
        await get_protect_client()
    except Exception as exc:
        logger.warning("Could not connect to Protect on startup: %s", exc)

    workers = [
        ("transcription", _supervised("transcription", transcription_worker)),
        ("compression", _supervised("compression", audio_compression_worker)),
        ("auto_sync", _supervised("auto_sync", auto_sync_worker)),
        ("auto_summary", _supervised("auto_summary", auto_summary_worker)),
        ("stuck_recovery", _supervised("stuck_recovery", stuck_processing_recovery_worker)),
        ("protect_health", _supervised("protect_health", protect_health_worker)),
    ]
    tasks: list[tuple[str, asyncio.Task]] = []
    for name, runner in workers:
        tasks.append((name, asyncio.create_task(runner(), name=name)))
        logger.info("Started worker: %s", name)

    yield

    # Shutdown
    for _, task in tasks:
        task.cancel()
    for name, task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            logger.info("%s worker cancelled", name)

    await close_protect_client()
    await close_clients()


app = FastAPI(
    title="Protect Transcribe",
    description="Speech transcription service for UniFi Protect",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)

# Static files (JS, CSS assets). The web app manifest needs its own MIME type.
mimetypes.add_type("application/manifest+json", ".webmanifest")
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Routers
for _router in (
    health.router,
    webhook.router,
    transcriptions.router,
    settings.router,
    summaries.router,
    sync.router,
    system.router,
    analytics.router,
    export.router,
    ws.router,
):
    app.include_router(_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
