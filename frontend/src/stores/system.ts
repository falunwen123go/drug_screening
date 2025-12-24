import { defineStore } from 'pinia';
import axios from '@/api/axios';
import { SystemInfo } from '@/types';

export const useSystemStore = defineStore('system', {
  state: () => ({
    info: null as SystemInfo | null,
    loading: false,
    error: null as string | null,
    loadingModel: false, // Track model switching state separately
  }),
  actions: {
    async fetchSystemInfo() {
      this.loading = true;
      try {
        const data = await axios.get<any, SystemInfo>('/system/info');
        this.info = data;
        this.error = null;
      } catch (err: any) {
        this.error = '无法连接到后端服务。';
        this.info = null;
      } finally {
        this.loading = false;
      }
    },

    async loadModel(modelName: string) {
        this.loadingModel = true;
        try {
            await axios.post('/system/load_model', { model_name: modelName });
            await this.fetchSystemInfo(); // Refresh info to update current model
            return true;
        } catch (err: any) {
            this.error = `加载模型 ${modelName} 失败: ${err.response?.data?.detail || err.message}`;
            return false;
        } finally {
            this.loadingModel = false;
        }
    }
  }
});