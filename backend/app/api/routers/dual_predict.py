from fastapi import APIRouter, HTTPException
import logging
import torch
import os

from app.models.schemas import (
    DualModelPredictionRequest, 
    DualModelPredictionResponse, 
    DualModelResult
)
from app.services.ml_service import ml_service
from app.services.chemistry import check_lipinski
from app.core.config import PROJECT_ROOT
from app.core.model_config import get_model_config

from features.molecular_features import MolecularFeaturizer
from models.drug_models import DrugPredictorMLPv2
from inference.predictor import DrugPredictor

router = APIRouter()
logger = logging.getLogger(__name__)

# 缓存的模型预测器
_model_cache = {}

def get_cached_predictor(model_name: str):
    """获取缓存的模型预测器，避免重复加载"""
    global _model_cache
    
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 推断模型结构
        hidden_dims = []
        layer_idx = 0
        while f'hidden_layers.{layer_idx}.weight' in state_dict:
            weight_shape = state_dict[f'hidden_layers.{layer_idx}.weight'].shape
            hidden_dims.append(weight_shape[0])
            layer_idx += 4
        
        input_dim = state_dict['hidden_layers.0.weight'].shape[1]
        output_dim = state_dict['output_layer.weight'].shape[0]
        
        model = DrugPredictorMLPv2(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=0.5,
            task_type='binary'
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        featurizer = MolecularFeaturizer(fingerprint_size=1024, radius=2)
        predictor = DrugPredictor(model, featurizer, device=device)
        
        _model_cache[model_name] = predictor
        logger.info(f"Cached model: {model_name}")
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None


@router.post("/predict/dual", response_model=DualModelPredictionResponse)
async def predict_dual_model(request: DualModelPredictionRequest):
    """
    使用两个模型同时预测：BBBP（血脑屏障穿透性）和ESOL（水溶性）
    """
    try:
        bbbp_result = None
        esol_result = None
        properties = None
        lipinski_passed = False
        
        # BBBP预测
        bbbp_predictor = get_cached_predictor('bbbp_model.pth')
        if bbbp_predictor:
            result = bbbp_predictor.predict_with_properties(request.smiles)
            if result['prediction'] is not None:
                config = get_model_config('bbbp_model.pth')
                score = result['prediction']
                label = config['high_label'] if score > config['threshold'] else config['low_label']
                bbbp_result = DualModelResult(
                    model_name="BBBP",
                    model_cn_name="血脑屏障穿透性",
                    score=float(score),
                    label=label,
                    icon="🧠",
                    unit=""
                )
                properties = result['properties']
                lipinski_passed = check_lipinski(properties)
        
        # ESOL预测
        esol_predictor = get_cached_predictor('esol_model.pth')
        if esol_predictor:
            result = esol_predictor.predict_with_properties(request.smiles)
            if result['prediction'] is not None:
                config = get_model_config('esol_model.pth')
                score = result['prediction']
                label = config['high_label'] if score > config['threshold'] else config['low_label']
                esol_result = DualModelResult(
                    model_name="ESOL",
                    model_cn_name="水溶性预测",
                    score=float(score),
                    label=label,
                    icon="💧",
                    unit="log mol/L"
                )
                if properties is None:
                    properties = result['properties']
                    lipinski_passed = check_lipinski(properties)
        
        if bbbp_result is None and esol_result is None:
            return DualModelPredictionResponse(
                smiles=request.smiles,
                bbbp_result=None,
                esol_result=None,
                properties=None,
                lipinski_passed=False,
                status="failed",
                error="无法加载任何模型或SMILES无效"
            )
        
        return DualModelPredictionResponse(
            smiles=request.smiles,
            bbbp_result=bbbp_result,
            esol_result=esol_result,
            properties=properties,
            lipinski_passed=lipinski_passed
        )
        
    except Exception as e:
        logger.error(f"Dual prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
