<template>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-emerald-600 flex items-center gap-3">
        <span class="text-4xl">🧪</span>
        单分子预测
      </h1>
      <p class="mt-2 text-gray-600">输入药物分子的SMILES字符串或使用分子编辑器绘制，进行活性预测和类药性评估</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- Left Panel: Input -->
      <div class="space-y-6">
        <div class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>📝</span> 输入分子
          </h2>
          
          <!-- Input Mode Tabs -->
          <div class="flex border-b border-gray-200 mb-4">
            <button 
              @click="inputMode = 'text'"
              :class="inputMode === 'text' ? 'border-emerald-500 text-emerald-600' : 'border-transparent text-gray-500 hover:text-gray-700'"
              class="px-4 py-2 text-sm font-medium border-b-2 transition-colors"
            >
              ✏️ 手动输入
            </button>
            <button 
              @click="inputMode = 'editor'"
              :class="inputMode === 'editor' ? 'border-emerald-500 text-emerald-600' : 'border-transparent text-gray-500 hover:text-gray-700'"
              class="px-4 py-2 text-sm font-medium border-b-2 transition-colors"
            >
              🎨 分子编辑器
            </button>
          </div>
          
          <div class="space-y-4">
            <!-- Text Input Mode -->
            <div v-if="inputMode === 'text'">
              <label for="smiles" class="block text-sm font-medium text-gray-700 mb-1">输入SMILES字符串</label>
              <div class="relative">
                <input 
                  type="text" 
                  v-model="smilesInput"
                  id="smiles" 
                  class="block w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-sm font-mono transition-all"
                  placeholder="例如：CC(=O)OC1=CC=CC=C1C(=O)O"
                />
                <button 
                  @click="smilesInput = ''"
                  v-if="smilesInput"
                  class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              
              <!-- Example Molecules -->
              <div class="mt-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">或选择示例分子</label>
                <select 
                  v-model="selectedExample" 
                  @change="onExampleSelect"
                  class="block w-full px-4 py-2.5 rounded-lg border border-gray-300 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 text-sm"
                >
                  <option value="">自定义</option>
                  <option v-for="(mol, idx) in exampleMolecules" :key="idx" :value="mol.smiles">
                    {{ mol.name }}
                  </option>
                </select>
              </div>
            </div>

            <!-- Molecule Editor Mode -->
            <div v-if="inputMode === 'editor'" class="space-y-3">
              <MoleculeEditor 
                ref="moleculeEditor"
                :height="350"
                :initial-smiles="smilesInput"
                @update:smiles="onEditorSmilesUpdate"
              />
              <p class="text-xs text-gray-500">
                💡 提示：在编辑器中绘制分子结构，点击"获取SMILES"按钮生成SMILES字符串
              </p>
            </div>

            <!-- Prediction Mode Selection -->
            <div class="mt-4 p-4 bg-gray-50 rounded-lg">
              <label class="block text-sm font-medium text-gray-700 mb-3">预测模式</label>
              <div class="flex gap-4">
                <label class="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="radio" 
                    v-model="predictionMode" 
                    value="single"
                    class="text-emerald-600 focus:ring-emerald-500"
                  />
                  <span class="text-sm">单模型预测</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="radio" 
                    v-model="predictionMode" 
                    value="dual"
                    class="text-emerald-600 focus:ring-emerald-500"
                  />
                  <span class="text-sm">🔥 双模型预测 (推荐)</span>
                </label>
              </div>
              <p class="text-xs text-gray-500 mt-2">
                {{ predictionMode === 'dual' ? '同时使用BBBP和ESOL模型，获得更全面的预测结果' : '使用当前加载的模型进行预测' }}
              </p>
            </div>

            <!-- Predict Button -->
            <button 
              @click="handlePredict" 
              :disabled="!smilesInput || drugStore.loading"
              class="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-lg text-white font-medium transition-all
                     bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600
                     disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
            >
              <svg v-if="drugStore.loading" class="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              <span>🔬 开始预测</span>
            </button>
            
            <!-- Success Message -->
            <div v-if="(drugStore.singleResult || drugStore.dualResult) && !drugStore.error" class="bg-emerald-50 border border-emerald-200 rounded-lg p-3 flex items-center gap-2 text-emerald-700">
              <span>✅</span>
              <span class="text-sm font-medium">预测完成!</span>
            </div>

            <!-- Error Message -->
            <div v-if="drugStore.error" class="bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-2 text-red-700">
              <span>❌</span>
              <span class="text-sm">{{ drugStore.error }}</span>
            </div>
          </div>
        </div>

        <!-- Dual Model Results -->
        <div v-if="drugStore.dualResult && predictionMode === 'dual'" class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">🔬 双模型预测结果</h3>
          
          <div class="grid grid-cols-2 gap-4">
            <!-- BBBP Result -->
            <div class="p-4 rounded-lg" :class="drugStore.dualResult.bbbp_result?.score && drugStore.dualResult.bbbp_result.score > 0.5 ? 'bg-emerald-50 border border-emerald-200' : 'bg-amber-50 border border-amber-200'">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">🧠</span>
                <span class="font-medium text-gray-900">血脑屏障穿透性</span>
              </div>
              <div class="text-3xl font-bold" :class="drugStore.dualResult.bbbp_result?.score && drugStore.dualResult.bbbp_result.score > 0.5 ? 'text-emerald-600' : 'text-amber-600'">
                {{ drugStore.dualResult.bbbp_result?.score?.toFixed(4) || 'N/A' }}
              </div>
              <p class="text-sm mt-1" :class="drugStore.dualResult.bbbp_result?.score && drugStore.dualResult.bbbp_result.score > 0.5 ? 'text-emerald-600' : 'text-amber-600'">
                {{ drugStore.dualResult.bbbp_result?.label || '无法预测' }}
              </p>
            </div>
            
            <!-- ESOL Result -->
            <div class="p-4 rounded-lg" :class="drugStore.dualResult.esol_result?.score && drugStore.dualResult.esol_result.score > -3 ? 'bg-blue-50 border border-blue-200' : 'bg-amber-50 border border-amber-200'">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">💧</span>
                <span class="font-medium text-gray-900">水溶性预测</span>
              </div>
              <div class="text-3xl font-bold" :class="drugStore.dualResult.esol_result?.score && drugStore.dualResult.esol_result.score > -3 ? 'text-blue-600' : 'text-amber-600'">
                {{ drugStore.dualResult.esol_result?.score?.toFixed(4) || 'N/A' }}
              </div>
              <p class="text-sm mt-1" :class="drugStore.dualResult.esol_result?.score && drugStore.dualResult.esol_result.score > -3 ? 'text-blue-600' : 'text-amber-600'">
                {{ drugStore.dualResult.esol_result?.label || '无法预测' }}
                <span class="text-gray-500">{{ drugStore.dualResult.esol_result?.unit }}</span>
              </p>
            </div>
          </div>
        </div>

        <!-- Single Model Prediction Score -->
        <div v-if="drugStore.singleResult && predictionMode === 'single'" class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <!-- Model Info Badge -->
          <div v-if="drugStore.singleResult.model_info" class="mb-4 flex items-center gap-2">
            <span class="text-2xl">{{ drugStore.singleResult.model_info.icon }}</span>
            <div>
              <span class="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-medium">
                {{ drugStore.singleResult.model_info.name }}
              </span>
              <span class="ml-2 text-sm text-gray-500">{{ drugStore.singleResult.model_info.cn_name }}</span>
            </div>
          </div>
          
          <h3 class="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">预测得分</h3>
          <div class="text-5xl font-bold" :class="scoreColor">
            {{ (drugStore.singleResult.prediction! * 1).toFixed(4) }}
            <span v-if="drugStore.singleResult.model_info?.unit" class="text-lg text-gray-500 ml-1">
              {{ drugStore.singleResult.model_info.unit }}
            </span>
          </div>
          <p class="text-sm text-gray-500 mt-2">
            {{ drugStore.singleResult.prediction_label }}
          </p>
          
          <!-- Progress Bar -->
          <div class="mt-4 h-3 bg-gray-200 rounded-full overflow-hidden">
            <div 
              class="h-full rounded-full transition-all duration-500"
              :class="progressBarColor"
              :style="{ width: progressBarWidth + '%' }"
            ></div>
          </div>
        </div>

        <!-- Properties -->
        <div v-if="currentProperties" class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">分子性质</h3>
          
          <div class="grid grid-cols-3 gap-4">
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">分子量</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.MolecularWeight?.toFixed(2) || 0 }} Da</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">LogP</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.LogP?.toFixed(2) || 0 }}</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">TPSA</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.TPSA?.toFixed(2) || 0 }} Ų</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">氢键供体</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.NumHDonors || 0 }}</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">氢键受体</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.NumHAcceptors || 0 }}</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
              <div class="text-xs text-gray-500 mb-1">旋转键数</div>
              <div class="text-xl font-bold text-gray-900">{{ currentProperties.NumRotatableBonds || 0 }}</div>
            </div>
          </div>

          <!-- Expandable full properties -->
          <details class="mt-4">
            <summary class="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium">
              &gt; 查看所有性质
            </summary>
            <div class="mt-3 bg-gray-50 rounded-lg p-4 text-sm">
              <div v-for="(value, key) in currentProperties" :key="key" class="flex justify-between py-1 border-b border-gray-200 last:border-0">
                <span class="text-gray-600">{{ key }}</span>
                <span class="font-mono font-medium">{{ typeof value === 'number' ? value.toFixed(4) : value }}</span>
              </div>
            </div>
          </details>
        </div>
      </div>

      <!-- Right Panel: Structure & Lipinski -->
      <div class="space-y-6">
        <!-- Molecule Structure -->
        <div class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <span>🔬</span> 分子结构
          </h2>
          <div class="flex justify-center items-center bg-gray-50 rounded-lg p-4 min-h-[300px]">
            <img 
              v-if="smilesInput"
              :src="`http://localhost:8000/molecule/image?smiles=${encodeURIComponent(smilesInput)}`" 
              :alt="smilesInput"
              class="max-w-full max-h-[280px] object-contain"
              @error="handleImageError"
            />
            <div v-else class="text-gray-400 text-center">
              <div class="text-4xl mb-2">🧬</div>
              <div>输入SMILES后显示分子结构</div>
            </div>
          </div>
        </div>

        <!-- Lipinski Rules -->
        <div v-if="currentProperties" class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">药物相似性评估 (Lipinski五规则)</h3>
          <p class="text-sm text-gray-500 mb-4">Lipinski五规则用于评估化合物的类药性，符合这些规则的化合物更可能成为口服药物。</p>
          
          <div class="space-y-3">
            <div v-for="rule in lipinskiRules" :key="rule.name" 
                 class="flex items-center justify-between p-3 rounded-lg"
                 :class="rule.passed ? 'bg-emerald-50' : 'bg-red-50'">
              <div class="flex items-center gap-3">
                <span class="text-lg">{{ rule.passed ? '✅' : '❌' }}</span>
                <span class="font-medium" :class="rule.passed ? 'text-emerald-700' : 'text-red-700'">
                  {{ rule.name }}
                </span>
              </div>
              <span class="text-sm text-gray-500">{{ rule.value }} {{ rule.condition }}</span>
            </div>
          </div>

          <div class="mt-4 p-4 rounded-lg" :class="currentLipinskiPassed ? 'bg-emerald-100' : 'bg-amber-100'">
            <div class="flex items-center gap-2">
              <span class="text-xl">{{ currentLipinskiPassed ? '🎉' : '⚠️' }}</span>
              <span class="font-semibold" :class="currentLipinskiPassed ? 'text-emerald-800' : 'text-amber-800'">
                {{ currentLipinskiPassed ? '该分子符合Lipinski五规则!' : '该分子不完全符合Lipinski五规则' }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useDrugStore } from '@/stores/drug';
import MoleculeEditor from '@/components/MoleculeEditor.vue';

const drugStore = useDrugStore();
const smilesInput = ref('CC(=O)OC1=CC=CC=C1C(=O)O');
const selectedExample = ref('');
const inputMode = ref<'text' | 'editor'>('text');
const predictionMode = ref<'single' | 'dual'>('dual');
const moleculeEditor = ref<InstanceType<typeof MoleculeEditor> | null>(null);

const exampleMolecules = [
  { name: 'Aspirin (阿司匹林)', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'Caffeine (咖啡因)', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { name: 'Ibuprofen (布洛芬)', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
  { name: 'Ethanol (乙醇)', smiles: 'CCO' },
  { name: 'Paracetamol (对乙酰氨基酚)', smiles: 'CC(=O)NC1=CC=C(C=C1)O' },
  { name: 'Dopamine (多巴胺)', smiles: 'NCCc1ccc(O)c(O)c1' },
];

onMounted(() => {
  drugStore.fetchSystemInfo();
});

const onExampleSelect = () => {
  if (selectedExample.value) {
    smilesInput.value = selectedExample.value;
  }
};

const onEditorSmilesUpdate = (smiles: string) => {
  smilesInput.value = smiles;
};

const handlePredict = () => {
  if (smilesInput.value) {
    if (predictionMode.value === 'dual') {
      drugStore.predictDual(smilesInput.value);
    } else {
      drugStore.predictSingle(smilesInput.value);
    }
  }
};

const handleImageError = (e: Event) => {
  (e.target as HTMLImageElement).style.display = 'none';
};

// 获取当前结果的属性
const currentProperties = computed(() => {
  if (predictionMode.value === 'dual' && drugStore.dualResult) {
    return drugStore.dualResult.properties;
  }
  return drugStore.singleResult?.properties;
});

const currentLipinskiPassed = computed(() => {
  if (predictionMode.value === 'dual' && drugStore.dualResult) {
    return drugStore.dualResult.lipinski_passed;
  }
  return drugStore.singleResult?.lipinski_passed;
});

const scoreColor = computed(() => {
  if (!drugStore.singleResult?.prediction) return 'text-gray-400';
  const modelInfo = drugStore.singleResult.model_info;
  const pred = drugStore.singleResult.prediction;
  
  if (modelInfo) {
    const threshold = modelInfo.threshold;
    if (modelInfo.task_type === 'binary') {
      return pred > threshold ? 'text-emerald-600' : 'text-amber-600';
    } else {
      // 回归任务（如ESOL），值越高越好
      return pred > threshold ? 'text-emerald-600' : 'text-amber-600';
    }
  }
  return pred > 0.5 ? 'text-emerald-600' : 'text-amber-600';
});

const progressBarColor = computed(() => {
  if (!drugStore.singleResult?.prediction) return 'bg-gray-400';
  const modelInfo = drugStore.singleResult.model_info;
  const pred = drugStore.singleResult.prediction;
  
  if (modelInfo && modelInfo.task_type === 'binary') {
    return pred > modelInfo.threshold ? 'bg-emerald-500' : 'bg-amber-500';
  }
  return pred > 0.5 ? 'bg-emerald-500' : 'bg-amber-500';
});

const progressBarWidth = computed(() => {
  if (!drugStore.singleResult?.prediction) return 0;
  const pred = drugStore.singleResult.prediction;
  const modelInfo = drugStore.singleResult.model_info;
  
  if (modelInfo && modelInfo.task_type === 'regression') {
    // 对于回归任务（如ESOL），归一化到0-100
    // ESOL范围大约是 -10 到 2
    return Math.max(0, Math.min(100, (pred + 10) / 12 * 100));
  }
  return pred * 100;
});

const lipinskiRules = computed(() => {
  const props = currentProperties.value;
  if (!props) return [];
  
  return [
    { name: '分子量 ≤ 500 Da', value: props.MolecularWeight?.toFixed(1), condition: '≤ 500', passed: (props.MolecularWeight || 0) <= 500 },
    { name: 'LogP ≤ 5', value: props.LogP?.toFixed(2), condition: '≤ 5', passed: (props.LogP || 0) <= 5 },
    { name: '氢键供体 ≤ 5', value: props.NumHDonors, condition: '≤ 5', passed: (props.NumHDonors || 0) <= 5 },
    { name: '氢键受体 ≤ 10', value: props.NumHAcceptors, condition: '≤ 10', passed: (props.NumHAcceptors || 0) <= 10 },
  ];
});
</script>