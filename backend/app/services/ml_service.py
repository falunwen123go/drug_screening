import torch
import os
import logging
import platform
from typing import Dict, Any, Optional, List

# Ensure sys.path is set
from backend.app.core.config import PROJECT_ROOT

from features.molecular_features import MolecularFeaturizer
from models.drug_models import DrugPredictorMLP
from inference.predictor import DrugPredictor

logger = logging.getLogger(__name__)

# Try importing psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class MLService:
    def __init__(self):
        self.predictor: Optional[DrugPredictor] = None
        self.device: str = "cpu"
        self.current_model_name: Optional[str] = None
        self.models_dir = os.path.join(PROJECT_ROOT, 'saved_models')

    def get_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def list_available_models(self) -> List[str]:
        """Lists all .pth files in the saved_models directory."""
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]

    def load_model(self, model_name: str = 'bbbp_model.pth') -> bool:
        """Loads a specific model by name."""
        try:
            self.device = self.get_device()
            logger.info(f"Loading model {model_name} on {self.device}...")
            
            model_path = os.path.join(self.models_dir, model_name)
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            # Model configuration
            # WARNING: Assuming a fixed architecture for now. 
            # Ideally, architecture config should be saved alongside weights (e.g., config.json).
            model = DrugPredictorMLP(input_dim=1024, hidden_dims=[512, 256, 128], output_dim=1)
            
            state_dict = torch.load(model_path, map_location=self.device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.error(f"State dict mismatch: {e}")
                # Fallback: try strict=False or different architecture if needed
                # For now, just fail
                return False

            model = model.to(self.device)
            model.eval()
            
            featurizer = MolecularFeaturizer(fingerprint_size=1024, radius=2)
            self.predictor = DrugPredictor(model, featurizer, device=self.device)
            self.current_model_name = model_name
            
            logger.info(f"Model {model_name} loaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.predictor = None
            self.current_model_name = None
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Returns detailed system and model information."""
        
        cpu_info = {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "version": platform.version()
        }
        
        memory_info = {}
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            memory_info = {
                "total": f"{mem.total / (1024**3):.2f} GB",
                "available": f"{mem.available / (1024**3):.2f} GB",
                "percent": f"{mem.percent}%"
            }
        
        gpu_info = None
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count()
            }

        return {
            "status": "healthy" if self.predictor else "no_model",
            "device": self.device,
            "current_model": self.current_model_name,
            "available_models": self.list_available_models(),
            "cpu_info": cpu_info,
            "memory_info": memory_info,
            "gpu_info": gpu_info
        }

ml_service = MLService()