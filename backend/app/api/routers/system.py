from fastapi import APIRouter, HTTPException
from app.services.ml_service import ml_service
from app.models.schemas import SystemInfoResponse, LoadModelRequest

router = APIRouter()

@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    return ml_service.get_system_info()

@router.post("/load_model")
async def load_model(request: LoadModelRequest):
    success = ml_service.load_model(request.model_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {request.model_name}")
    return {"status": "success", "message": f"Model {request.model_name} loaded successfully", "current_model": request.model_name}
