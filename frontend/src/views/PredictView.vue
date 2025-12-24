<template>
  <div class="space-y-6">
    <div class="bg-white shadow rounded-lg p-6">
      <h2 class="text-xl font-bold text-gray-900 mb-4">单分子属性预测</h2>
      
      <div class="space-y-4">
        <div>
          <label for="smiles" class="block text-sm font-medium text-gray-700">SMILES 字符串</label>
          <div class="mt-1 flex rounded-md shadow-sm">
            <input 
              type="text" 
              v-model="smilesInput"
              id="smiles" 
              class="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm" 
              placeholder="例如：CC(=O)OC1=CC=CC=C1C(=O)O"
            />
          </div>
          <p class="mt-2 text-sm text-gray-500">请输入合法的药物分子 SMILES 表达式。</p>
        </div>

        <button 
          @click="handlePredict" 
          :disabled="!smilesInput || drugStore.loading"
          class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          <span v-if="drugStore.loading" class="animate-spin mr-2">↻</span>
          开始预测
        </button>
        
        <div v-if="drugStore.error" class="text-red-600 text-sm mt-2">
          {{ drugStore.error }}
        </div>
      </div>
    </div>

    <!-- Results Section -->
    <div v-if="drugStore.singleResult" class="bg-white shadow rounded-lg p-6 border-t-4" :class="drugStore.singleResult.prediction && drugStore.singleResult.prediction > 0.5 ? 'border-green-500' : 'border-yellow-500'">
      <h3 class="text-lg font-medium text-gray-900 mb-4">分析结果</h3>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Main Prediction -->
        <div>
          <h4 class="text-sm font-semibold text-gray-500 uppercase tracking-wider">活性/渗透概率</h4>
          <div class="mt-1 flex items-baseline">
            <span class="text-3xl font-extrabold text-gray-900">
              {{ (drugStore.singleResult.prediction! * 100).toFixed(1) }}%
            </span>
            <span class="ml-2 px-2 py-0.5 rounded text-xs font-medium" 
              :class="drugStore.singleResult.prediction! > 0.5 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'">
              {{ drugStore.singleResult.prediction! > 0.5 ? '高概率' : '低概率' }}
            </span>
          </div>
          
          <div class="mt-4">
            <h4 class="text-sm font-semibold text-gray-500 uppercase tracking-wider">Lipinski 五规则</h4>
             <div class="mt-1 flex items-center">
                <span class="text-xl mr-2">{{ drugStore.singleResult.lipinski_passed ? '✅' : '⚠️' }}</span>
                <span class="text-gray-700">{{ drugStore.singleResult.lipinski_passed ? '通过' : '未通过' }}</span>
             </div>
          </div>
        </div>

        <!-- Molecular Properties -->
        <div class="bg-gray-50 rounded p-4">
          <h4 class="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">物理化学属性</h4>
          <dl class="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <div class="col-span-1 text-gray-600">分子量 (MW):</div>
            <div class="col-span-1 font-medium">{{ drugStore.singleResult.properties?.MolecularWeight.toFixed(2) }}</div>
            
            <div class="col-span-1 text-gray-600">脂水分配系数 (LogP):</div>
            <div class="col-span-1 font-medium">{{ drugStore.singleResult.properties?.LogP.toFixed(2) }}</div>
            
            <div class="col-span-1 text-gray-600">拓扑极性表面积 (TPSA):</div>
            <div class="col-span-1 font-medium">{{ drugStore.singleResult.properties?.TPSA.toFixed(2) }}</div>
            
            <div class="col-span-1 text-gray-600">氢键供体 (H-Donors):</div>
            <div class="col-span-1 font-medium">{{ drugStore.singleResult.properties?.NumHDonors }}</div>
            
            <div class="col-span-1 text-gray-600">氢键受体 (H-Acceptors):</div>
            <div class="col-span-1 font-medium">{{ drugStore.singleResult.properties?.NumHAcceptors }}</div>
          </dl>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useDrugStore } from '@/stores/drug';

const drugStore = useDrugStore();
const smilesInput = ref('CC(=O)OC1=CC=CC=C1C(=O)O'); // Default Aspirin

const handlePredict = () => {
  if (smilesInput.value) {
    drugStore.predictSingle(smilesInput.value);
  }
};
</script>