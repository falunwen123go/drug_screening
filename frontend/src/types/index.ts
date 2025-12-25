export interface MoleculeProperties {
  MolecularWeight: number;
  LogP: number;
  TPSA: number;
  NumHDonors: number;
  NumHAcceptors: number;
  NumRotatableBonds: number;
  [key: string]: number;
}

export interface SinglePredictionResponse {
  smiles: string;
  prediction: number | null;
  prediction_label: string;
  properties: MoleculeProperties | null;
  lipinski_passed: boolean;
  status: string;
  error?: string;
}

export interface ScreenedMolecule {
  rank: number;
  smiles: string;
  score: number;
  properties: MoleculeProperties;
}

export interface BatchScreeningResponse {
  total_input: number;
  total_screened: number;
  results: ScreenedMolecule[];
}

export interface CpuInfo {
  processor: string;
  machine: string;
  system: string;
  version: string;
}

export interface MemoryInfo {
  total: string;
  available: string;
  percent: string;
}

export interface GpuInfo {
  name: string;
  count: number;
}

export interface SystemInfo {
  status: string;
  device: string;
  current_model: string | null;
  available_models: string[];
  cpu_info: CpuInfo;
  memory_info: MemoryInfo;
  gpu_info: GpuInfo | null;
  model_loaded?: boolean; // For backward compatibility if needed, though we rely on current_model
}

export interface SystemHealth {
    status: string;
    device: string;
    model_loaded: boolean;
}