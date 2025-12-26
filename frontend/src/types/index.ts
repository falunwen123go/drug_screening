export interface MoleculeProperties {
  MolecularWeight: number;
  LogP: number;
  TPSA: number;
  NumHDonors: number;
  NumHAcceptors: number;
  NumRotatableBonds: number;
  [key: string]: number;
}

export interface ModelInfo {
  name: string;
  full_name: string;
  cn_name: string;
  description: string;
  task_type: string;
  score_meaning: string;
  high_label: string;
  low_label: string;
  threshold: number;
  unit: string;
  icon: string;
}

export interface SinglePredictionResponse {
  smiles: string;
  prediction: number | null;
  prediction_label: string;
  properties: MoleculeProperties | null;
  lipinski_passed: boolean;
  model_info: ModelInfo | null;
  status: string;
  error?: string;
}

// 双模型预测类型
export interface DualModelResult {
  model_name: string;
  model_cn_name: string;
  score: number | null;
  label: string;
  icon: string;
  unit: string;
}

export interface DualModelPredictionResponse {
  smiles: string;
  bbbp_result: DualModelResult | null;
  esol_result: DualModelResult | null;
  properties: MoleculeProperties | null;
  lipinski_passed: boolean;
  status: string;
  error?: string;
}

export interface ScreenedMolecule {
  rank: number;
  smiles: string;
  score: number;
  bbbp_score?: number;
  esol_score?: number;
  properties: MoleculeProperties;
}

export interface BatchScreeningResponse {
  total_input: number;
  total_screened: number;
  current_model?: string;
  use_dual_model?: boolean;
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