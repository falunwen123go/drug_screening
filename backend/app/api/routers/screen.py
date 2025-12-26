from fastapi import APIRouter, HTTPException
import logging
import os
import torch

# Ensure sys.path is set
from app.core.config import PROJECT_ROOT 

from app.models.schemas import BatchScreeningRequest, BatchScreeningResponse, ScreenedMolecule
from app.services.ml_service import ml_service
from inference.predictor import DrugScreener, DrugPredictor
from features.molecular_features import MolecularFeaturizer
from models.drug_models import DrugPredictorMLPv2

router = APIRouter()
logger = logging.getLogger(__name__)

# Model cache for dual model screening
_model_cache = {}

def get_predictor(model_name: str) -> DrugPredictor:
    """Get or create a cached predictor for the given model."""
    if model_name not in _model_cache:
        models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found at {model_path}")
        
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
            logger.info(f"Loaded and cached model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return _model_cache[model_name]

@router.post("/screen", response_model=BatchScreeningResponse)
async def screen_library(request: BatchScreeningRequest):
    try:
        if request.use_dual_model:
            # Dual model screening - use both BBBP and ESOL
            return await screen_with_dual_model(request)
        else:
            # Single model screening
            return await screen_with_single_model(request)
    except Exception as e:
        logger.error(f"Screening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def screen_with_single_model(request: BatchScreeningRequest):
    """Screen with the currently loaded model."""
    if not ml_service.predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    screener = DrugScreener(ml_service.predictor)
    
    # 1. Screen (Predict & Sort)
    results_df = screener.screen_library(
        request.smiles_list, 
        top_k=request.top_k if not request.apply_lipinski else len(request.smiles_list),
        ascending=request.ascending
    )
    
    # 2. Filter (Optional)
    if request.apply_lipinski:
        results_df = screener.filter_by_lipinski(results_df)
        results_df = results_df.head(request.top_k)
    
    # 3. Format Response
    output_results = []
    for idx, row in results_df.iterrows():
        props = {k: v for k, v in row.to_dict().items() if k not in ['smiles', 'score', 'rank']}
        
        output_results.append(ScreenedMolecule(
            rank=idx + 1 if 'rank' not in row else row['rank'],
            smiles=row['smiles'],
            score=float(row['score']),
            properties=props
        ))
        
    return BatchScreeningResponse(
        total_input=len(request.smiles_list),
        total_screened=len(output_results),
        current_model=ml_service.current_model,
        use_dual_model=False,
        results=output_results
    )

async def screen_with_dual_model(request: BatchScreeningRequest):
    """Screen with both BBBP and ESOL models."""
    # Load both models
    bbbp_predictor = get_predictor("bbbp_model.pth")
    esol_predictor = get_predictor("esol_model.pth")
    
    bbbp_screener = DrugScreener(bbbp_predictor)
    esol_screener = DrugScreener(esol_predictor)
    
    # Get BBBP scores (primary ranking)
    bbbp_df = bbbp_screener.screen_library(
        request.smiles_list, 
        top_k=len(request.smiles_list),
        ascending=False  # Higher is better for BBBP
    )
    
    # Get ESOL scores
    esol_df = esol_screener.screen_library(
        request.smiles_list, 
        top_k=len(request.smiles_list),
        ascending=False  # Higher (less negative) is better for ESOL
    )
    
    # Create a lookup for ESOL scores
    esol_scores = dict(zip(esol_df['smiles'], esol_df['score']))
    
    # Merge results
    bbbp_df['bbbp_score'] = bbbp_df['score']
    bbbp_df['esol_score'] = bbbp_df['smiles'].map(esol_scores)
    
    # Apply Lipinski filter if requested
    if request.apply_lipinski:
        bbbp_df = bbbp_screener.filter_by_lipinski(bbbp_df)
    
    # Sort by BBBP score and get top_k
    bbbp_df = bbbp_df.sort_values('bbbp_score', ascending=request.ascending).head(request.top_k)
    
    # Format Response
    output_results = []
    for idx, row in bbbp_df.iterrows():
        props = {k: v for k, v in row.to_dict().items() 
                 if k not in ['smiles', 'score', 'rank', 'bbbp_score', 'esol_score']}
        
        output_results.append(ScreenedMolecule(
            rank=len(output_results) + 1,
            smiles=row['smiles'],
            score=float(row['bbbp_score']),  # Primary score is BBBP
            bbbp_score=float(row['bbbp_score']),
            esol_score=float(row['esol_score']) if row['esol_score'] is not None else None,
            properties=props
        ))
        
    return BatchScreeningResponse(
        total_input=len(request.smiles_list),
        total_screened=len(output_results),
        use_dual_model=True,
        results=output_results
    )
