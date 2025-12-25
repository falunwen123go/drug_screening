<template>
  <div class="space-y-6">
    <div class="bg-white shadow rounded-lg p-6">
      <h2 class="text-xl font-bold text-gray-900 mb-4">批量虚拟筛选</h2>
      
      <div class="space-y-4">
        <div>
          <label for="smiles-list" class="block text-sm font-medium text-gray-700">SMILES 列表 (每行一个)</label>
          <textarea 
            v-model="smilesText"
            id="smiles-list" 
            rows="6"
            class="mt-1 block w-full px-3 py-2 rounded-md border border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm font-mono"
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O&#10;CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
          ></textarea>
        </div>

        <div class="flex items-center space-x-6">
            <div>
                <label for="top-k" class="block text-sm font-medium text-gray-700">筛选 Top K 候选</label>
                <input type="number" v-model.number="topK" id="top-k" class="mt-1 block w-24 px-3 py-2 rounded-md border border-gray-300 sm:text-sm" min="1" max="100">
            </div>
            
            <div class="flex items-center h-full pt-6">
                <input id="lipinski" type="checkbox" v-model="applyLipinski" class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded">
                <label for="lipinski" class="ml-2 block text-sm text-gray-900">
                    应用 Lipinski 规则过滤
                </label>
            </div>
        </div>

        <button 
          @click="handleScreen" 
          :disabled="!smilesText || drugStore.loading"
          class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          <span v-if="drugStore.loading" class="animate-spin mr-2">↻</span>
          开始筛选
        </button>

         <div v-if="drugStore.error" class="text-red-600 text-sm mt-2">
          {{ drugStore.error }}
        </div>
      </div>
    </div>

    <!-- Results Table -->
    <div v-if="drugStore.screeningResults" class="bg-white shadow rounded-lg overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200 bg-gray-50 flex justify-between items-center">
        <h3 class="text-lg font-medium text-gray-900">筛选结果</h3>
        <span class="text-sm text-gray-500">
            共处理 {{ drugStore.screeningResults.total_input }} 个分子，筛选出 {{ drugStore.screeningResults.total_screened }} 个
        </span>
      </div>
      
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">排名</th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">评分</th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">SMILES</th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">分子量 (MW)</th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">LogP</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="mol in drugStore.screeningResults.results" :key="mol.rank">
              <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">#{{ mol.rank }}</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-bold text-indigo-600">{{ mol.score.toFixed(4) }}</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono truncate max-w-xs" :title="mol.smiles">{{ mol.smiles }}</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ mol.properties.MolecularWeight.toFixed(1) }}</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ mol.properties.LogP.toFixed(2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <div v-if="drugStore.screeningResults.results.length === 0" class="p-6 text-center text-gray-500">
        没有找到符合条件的候选分子。
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useDrugStore } from '@/stores/drug';

const drugStore = useDrugStore();
const smilesText = ref('CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCCO');
const topK = ref(10);
const applyLipinski = ref(true);

const handleScreen = () => {
    const smilesList = smilesText.value
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0);
        
    if (smilesList.length > 0) {
        drugStore.screenLibrary(smilesList, topK.value, applyLipinski.value);
    }
};
</script>