from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app import jobs
from app import summaries as summary_service
from app.auth import require_api_token
from app.summaries import PERIODS

router = APIRouter()


@router.get("/api/summaries")
async def get_summaries(period: str = Query("daily"), limit: int = Query(60, ge=1, le=2000)):
    if period not in PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {', '.join(PERIODS)}")
    data = summary_service.get_summaries(period, limit=limit)
    busy = {j["period_key"] for j in jobs.running("summary") if j.get("period_type") == period}
    for item in data["items"]:
        item["generating"] = item["period_key"] in busy
    return data


@router.post("/api/summaries/generate", dependencies=[Depends(require_api_token)])
async def generate_summary(request: Request):
    data = await request.json()
    period = data.get("period_type", "daily")
    period_key = data.get("period_key", "")

    if period not in PERIODS:
        raise HTTPException(
            status_code=400, detail=f"period_type must be one of {', '.join(PERIODS)}"
        )
    if not period_key:
        raise HTTPException(status_code=400, detail="period_key is required")

    # Runs in the background (a month can take minutes); poll GET /api/jobs/{id}.
    for job in jobs.running("summary"):
        if job.get("period_type") == period and job.get("period_key") == period_key:
            return job
    return jobs.start(
        "summary",
        summary_service.generate_summary(period, period_key),
        period_type=period,
        period_key=period_key,
    )
