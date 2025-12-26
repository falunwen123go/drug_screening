from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import SinglePredictionRequest, SinglePredictionResponse, ModelInfo
from app.services.ml_service import ml_service
from app.services.chemistry import check_lipinski
from app.core.model_config import get_model_config

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    if not ml_service.predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = ml_service.predictor.predict_with_properties(request.smiles)
        
        # 获取当前模型配置
        model_config = get_model_config(ml_service.current_model_name or "unknown")
        model_info = ModelInfo(**model_config)
        
        if result['prediction'] is None:
             return SinglePredictionResponse(
                smiles=request.smiles,
                prediction=None,
                prediction_label="Error",
                properties=None,
                lipinski_passed=False,
                model_info=model_info,
                status="failed",
                error="Invalid SMILES or prediction failed"
            )

        props = result['properties']
        passed_lipinski = check_lipinski(props)
        
        # 根据模型配置生成标签
        pred_value = result['prediction']
        if model_config['task_type'] == 'binary':
            label = model_config['high_label'] if pred_value > model_config['threshold'] else model_config['low_label']
        else:
            # 回归任务
            label = model_config['high_label'] if pred_value > model_config['threshold'] else model_config['low_label']
        
        return SinglePredictionResponse(
            smiles=request.smiles,
            prediction=float(pred_value),
            prediction_label=label,
            properties=props,
            lipinski_passed=passed_lipinski,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
