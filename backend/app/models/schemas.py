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

class SinglePredictionResponse(BaseModel):
    smiles: str
    prediction: Optional[float]
    prediction_label: str 
    properties: Optional[Dict[str, float]]
    lipinski_passed: bool
    status: str = "success"
    error: Optional[str] = None

class BatchScreeningRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings to screen")
    top_k: int = Field(10, ge=1, le=1000, description="Number of top candidates to return")
    ascending: bool = Field(False, description="Sort order: True for lower scores first, False for higher scores first")
    apply_lipinski: bool = Field(True, description="Whether to filter results based on Lipinski's Rule of Five")

class ScreenedMolecule(BaseModel):
    rank: int
    smiles: str
    score: float
    properties: Optional[Dict[str, float]]

class BatchScreeningResponse(BaseModel):
    total_input: int
    total_screened: int
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