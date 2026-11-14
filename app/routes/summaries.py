from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app import summaries as summary_service
from app.auth import require_api_token
from app.summaries import PERIODS

router = APIRouter()


@router.get("/api/summaries")
async def get_summaries(period: str = Query("daily")):
    if period not in PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {', '.join(PERIODS)}")
    return summary_service.get_summaries(period)


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

    return await summary_service.generate_summary(period, period_key)
