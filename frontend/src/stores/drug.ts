import { defineStore } from 'pinia';
import axios from '@/api/axios';
import { SinglePredictionResponse, BatchScreeningResponse } from '@/types';

export const useDrugStore = defineStore('drug', {
  state: () => ({
    singleResult: null as SinglePredictionResponse | null,
    screeningResults: null as BatchScreeningResponse | null,
    loading: false,
    error: null as string | null,
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

    async screenLibrary(smilesList: string[], topK: number = 10, applyLipinski: boolean = true) {
      this.loading = true;
      this.error = null;
      this.screeningResults = null;
      try {
        const data = await axios.post<any, BatchScreeningResponse>('/screen', {
          smiles_list: smilesList,
          top_k: topK,
          ascending: false,
          apply_lipinski: applyLipinski
        });
        this.screeningResults = data;
      } catch (err: any) {
        this.error = err.response?.data?.detail || 'Screening failed.';
      } finally {
        this.loading = false;
      }
    }
  }
});
