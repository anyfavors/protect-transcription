"""
Tiny in-memory background job registry for slow LLM work (summaries, ask).

Requests through Cloudflare time out after ~100 s, so slow work runs as a task
and the browser polls GET /api/jobs/{id}. Jobs are lost on restart, which is
fine: the caller can simply start again.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

_MAX_JOBS = 100
_jobs: dict[str, dict[str, Any]] = {}
_tasks: set[asyncio.Task] = set()


def start(kind: str, coro: Coroutine[Any, Any, Any], **meta: Any) -> dict[str, Any]:
    """Run *coro* in the background and return the job record."""
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "id": job_id,
        "kind": kind,
        "status": "running",
        "result": None,
        "error": None,
        "started_at": time.time(),
        **meta,
    }
    _jobs[job_id] = job
    while len(_jobs) > _MAX_JOBS:
        _jobs.pop(next(iter(_jobs)))

    async def _run() -> None:
        try:
            job["result"] = await coro
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = getattr(exc, "detail", None) or str(exc)
            logger.warning("Job %s (%s) failed: %s", job_id, kind, job["error"])
        job["finished_at"] = time.time()

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return job


def get(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def running(kind: str) -> list[dict[str, Any]]:
    return [j for j in _jobs.values() if j["kind"] == kind and j["status"] == "running"]
