from fastapi import APIRouter
from app.services.ml_service import ml_service

router = APIRouter()

@router.get("/health")
async def health_check():
    status = "healthy" if ml_service.predictor is not None else "degraded"
    return {
        "status": status,
        "device": ml_service.device,
        "model_loaded": ml_service.predictor is not None
    }
