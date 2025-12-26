import { defineStore } from 'pinia';
import axios from '@/api/axios';
import { SinglePredictionResponse, BatchScreeningResponse, DualModelPredictionResponse } from '@/types';

export const useDrugStore = defineStore('drug', {
  state: () => ({
    singleResult: null as SinglePredictionResponse | null,
    dualResult: null as DualModelPredictionResponse | null,
    screeningResults: null as BatchScreeningResponse | null,
    loading: false,
    error: null as string | null,
    currentModel: 'bbbp_model.pth' as string,
    availableModels: [] as string[],
  }),
  actions: {
    async predictSingle(smiles: string) {
      this.loading = true;
      this.error = null;
      this.singleResult = null;
      try {
        const data = await axios.post<any, SinglePredictionResponse>('/predict', { smiles });
        this.singleResult = data;
      } catch (err: any) {
        this.error = err.response?.data?.detail || 'Prediction failed. Check SMILES validity.';
      } finally {
        this.loading = false;
      }
    },

    async predictDual(smiles: string) {
      this.loading = true;
      this.error = null;
      this.dualResult = null;
      try {
        const data = await axios.post<any, DualModelPredictionResponse>('/predict/dual', { smiles });
        this.dualResult = data;
      } catch (err: any) {
        this.error = err.response?.data?.detail || 'Dual prediction failed.';
      } finally {
        this.loading = false;
      }
    },

    async screenLibrary(smilesList: string[], topK: number = 10, applyLipinski: boolean = true, ascending: boolean = false, useDualModel: boolean = false) {
      this.loading = true;
      this.error = null;
      this.screeningResults = null;
      try {
        const data = await axios.post<any, BatchScreeningResponse>('/screen', {
          smiles_list: smilesList,
          top_k: topK,
          ascending: ascending,
          apply_lipinski: applyLipinski,
          use_dual_model: useDualModel
        });
        this.screeningResults = data;
      } catch (err: any) {
        this.error = err.response?.data?.detail || 'Screening failed.';
      } finally {
        this.loading = false;
      }
    },

    async loadModel(modelName: string) {
      this.loading = true;
      this.error = null;
      try {
        await axios.post('/system/load_model', { model_name: modelName });
        this.currentModel = modelName;
      } catch (err: any) {
        this.error = err.response?.data?.detail || 'Failed to load model.';
      } finally {
        this.loading = false;
      }
    },

    async fetchSystemInfo() {
      try {
        const data = await axios.get<any, any>('/system/info');
        this.currentModel = data.current_model || 'bbbp_model.pth';
        this.availableModels = data.available_models || [];
      } catch (err) {
        console.error('Failed to fetch system info');
      }
    }
  }
});
