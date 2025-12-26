from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class SinglePredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule", example="CC(=O)OC1=CC=CC=C1C(=O)O")

class MoleculeProperties(BaseModel):
    MolecularWeight: float
    LogP: float
    TPSA: float
    NumHDonors: int
    NumHAcceptors: int
    NumRotatableBonds: int
    class Config:
        extra = "allow"

class ModelInfo(BaseModel):
    """当前加载模型的信息"""
    name: str
    full_name: str
    cn_name: str
    description: str
    task_type: str
    score_meaning: str
    high_label: str
    low_label: str
    threshold: float
    unit: str
    icon: str
    
    class Config:
        protected_namespaces = ()

class SinglePredictionResponse(BaseModel):
    smiles: str
    prediction: Optional[float]
    prediction_label: str 
    properties: Optional[Dict[str, float]]
    lipinski_passed: bool
    model_info: Optional[ModelInfo] = None
    status: str = "success"
    error: Optional[str] = None
    
    class Config:
        protected_namespaces = ()

# 双模型预测请求和响应
class DualModelPredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")

class DualModelResult(BaseModel):
    """单个模型的预测结果"""
    model_name: str
    model_cn_name: str
    score: Optional[float]
    label: str
    icon: str
    unit: str
    
    class Config:
        protected_namespaces = ()

class DualModelPredictionResponse(BaseModel):
    smiles: str
    bbbp_result: Optional[DualModelResult]
    esol_result: Optional[DualModelResult]
    properties: Optional[Dict[str, float]]
    lipinski_passed: bool
    status: str = "success"
    error: Optional[str] = None

class BatchScreeningRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings to screen")
    top_k: int = Field(10, ge=1, le=1000, description="Number of top candidates to return")
    ascending: bool = Field(False, description="Sort order: True for lower scores first, False for higher scores first")
    apply_lipinski: bool = Field(True, description="Whether to filter results based on Lipinski's Rule of Five")
    use_dual_model: bool = Field(False, description="Whether to use both BBBP and ESOL models for scoring")

class ScreenedMolecule(BaseModel):
    rank: int
    smiles: str
    score: float
    bbbp_score: Optional[float] = None
    esol_score: Optional[float] = None
    properties: Optional[Dict[str, float]]

class BatchScreeningResponse(BaseModel):
    total_input: int
    total_screened: int
    current_model: Optional[str] = None
    use_dual_model: bool = False
    results: List[ScreenedMolecule]

class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model file to load (e.g., 'bbbp_model.pth')")

class SystemInfoResponse(BaseModel):
    status: str
    device: str
    current_model: Optional[str]
    available_models: List[str]
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    gpu_info: Optional[Dict[str, Any]] = None