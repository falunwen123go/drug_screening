<template>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-indigo-600 flex items-center gap-3">
        <span class="text-4xl">📊</span>
        批量筛选
      </h1>
      <p class="mt-2 text-gray-600">上传包含SMILES的CSV文件进行批量筛选</p>
    </div>

    <!-- Input Section -->
    <div class="bg-white shadow-lg rounded-xl p-6 border border-gray-100 mb-6">
      <!-- File Upload -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-2">选择CSV文件</label>
        <div 
          class="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-indigo-400 transition-colors cursor-pointer"
          :class="{ 'border-indigo-500 bg-indigo-50': isDragging }"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleFileDrop"
          @click="triggerFileInput"
        >
          <input 
            ref="fileInput"
            type="file" 
            accept=".csv"
            class="hidden"
            @change="handleFileSelect"
          />
          <div class="text-gray-400 mb-2">
            <svg class="mx-auto h-12 w-12" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </div>
          <p class="text-gray-600">Drag and drop file here</p>
          <p class="text-sm text-gray-400">Limit 200MB per file • CSV</p>
          <button class="mt-4 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50">
            Browse files
          </button>
        </div>
        
        <!-- Uploaded File Info -->
        <div v-if="uploadedFile" class="mt-4 flex items-center justify-between bg-gray-50 rounded-lg p-3">
          <div class="flex items-center gap-3">
            <span class="text-2xl">📄</span>
            <div>
              <p class="font-medium text-gray-900">{{ uploadedFile.name }}</p>
              <p class="text-sm text-gray-500">{{ formatFileSize(uploadedFile.size) }}</p>
            </div>
          </div>
          <button @click="removeFile" class="text-gray-400 hover:text-red-500 text-xl">×</button>
        </div>
      </div>

      <!-- Success Alert for File Load -->
      <div v-if="csvData.length > 0" class="mb-6 bg-emerald-50 border border-emerald-200 rounded-lg p-4 flex items-center gap-3">
        <span class="text-emerald-500 text-xl">✅</span>
        <span class="text-emerald-700 font-medium">已加载 {{ csvData.length }} 个化合物</span>
      </div>

      <!-- Data Preview Table -->
      <div v-if="csvData.length > 0" class="mb-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-3">数据预览</h3>
        <div class="overflow-x-auto border rounded-lg max-h-64">
          <table class="min-w-full divide-y divide-gray-200 text-sm">
            <thead class="bg-gray-50 sticky top-0">
              <tr>
                <th v-for="col in csvColumns" :key="col" class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  {{ col }}
                </th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="(row, idx) in csvData.slice(0, 10)" :key="idx" class="hover:bg-gray-50">
                <td v-for="col in csvColumns" :key="col" class="px-4 py-2 whitespace-nowrap truncate max-w-xs" :class="col === 'smiles' ? 'font-mono text-indigo-600' : ''">
                  {{ row[col] }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-if="csvData.length > 10" class="text-sm text-gray-500 mt-2">显示前 10 条，共 {{ csvData.length }} 条</p>
      </div>

      <!-- Manual Input (Alternative) -->
      <details class="mb-6">
        <summary class="cursor-pointer text-sm text-indigo-600 hover:text-indigo-700 font-medium">
          📝 或手动输入SMILES列表
        </summary>
        <div class="mt-3">
          <textarea 
            v-model="smilesText"
            rows="4"
            class="block w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm font-mono"
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O&#10;CN1C=NC2=C1C(=O)N(C(=O)N2C)C&#10;CCO"
          ></textarea>
        </div>
      </details>

      <!-- SMILES Column Selection -->
      <div v-if="csvColumns.length > 0" class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-2">选择SMILES列</label>
        <select 
          v-model="selectedSmilesColumn"
          class="block w-full px-4 py-2.5 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm"
        >
          <option v-for="col in csvColumns" :key="col" :value="col">{{ col }}</option>
        </select>
      </div>

      <!-- Screening Parameters -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-2">Top-K候选数</label>
          <div class="flex items-center gap-2">
            <input 
              type="number" 
              v-model.number="topK" 
              min="1" 
              max="1000"
              class="block w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm"
            />
            <button @click="topK = Math.max(1, topK - 1)" class="px-3 py-2 bg-gray-100 rounded-lg hover:bg-gray-200">−</button>
            <button @click="topK = Math.min(1000, topK + 1)" class="px-3 py-2 bg-gray-100 rounded-lg hover:bg-gray-200">+</button>
          </div>
        </div>
        
        <div class="flex items-center pt-6">
          <input 
            id="ascending" 
            type="checkbox" 
            v-model="ascending"
            class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label for="ascending" class="ml-2 text-sm text-gray-700">分数越小越好</label>
        </div>
        
        <div class="flex items-center pt-6">
          <input 
            id="lipinski" 
            type="checkbox" 
            v-model="applyLipinski"
            class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label for="lipinski" class="ml-2 text-sm text-gray-700">应用Lipinski过滤</label>
        </div>
        
        <div class="flex items-center pt-6">
          <input 
            id="dualModel" 
            type="checkbox" 
            v-model="useDualModel"
            class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label for="dualModel" class="ml-2 text-sm text-gray-700">🔥 双模型评分</label>
        </div>
      </div>
      
      <!-- Dual Model Info -->
      <div v-if="useDualModel" class="mb-6 p-4 bg-indigo-50 rounded-lg border border-indigo-100">
        <p class="text-sm text-indigo-700">
          <span class="font-medium">双模型模式：</span>
          同时使用 BBBP (血脑屏障穿透性) 和 ESOL (水溶性) 模型评分，结果将显示两个模型的得分。
          <br>
          <span class="text-indigo-500">排序依据：BBBP分数（用于筛选CNS药物）</span>
        </p>
      </div>

      <!-- Screen Button -->
      <button 
        @click="handleScreen" 
        :disabled="(!smilesText && csvData.length === 0) || drugStore.loading"
        class="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-lg text-white font-medium transition-all
               bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600
               disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
      >
        <svg v-if="drugStore.loading" class="animate-spin h-5 w-5" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
        <span>🔬 开始筛选</span>
      </button>

      <!-- Error Message -->
      <div v-if="drugStore.error" class="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-2 text-red-700">
        <span>❌</span>
        <span class="text-sm">{{ drugStore.error }}</span>
      </div>

      <!-- Success Message -->
      <div v-if="drugStore.screeningResults" class="mt-4 bg-emerald-50 border border-emerald-200 rounded-lg p-3 flex items-center gap-2 text-emerald-700">
        <span>✅</span>
        <span class="text-sm font-medium">筛选完成！找到 {{ drugStore.screeningResults.total_screened }} 个候选化合物</span>
      </div>
    </div>

    <!-- Results Section -->
    <div v-if="drugStore.screeningResults" class="space-y-6">
      <!-- Results Table -->
      <div class="bg-white shadow-lg rounded-xl overflow-hidden border border-gray-100">
        <div class="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-indigo-500 to-purple-500 flex justify-between items-center">
          <h3 class="text-lg font-semibold text-white">筛选结果</h3>
          <div class="flex items-center gap-4">
            <span class="text-sm text-white/80">
              输入 {{ drugStore.screeningResults.total_input }} 个 → 筛选出 {{ drugStore.screeningResults.total_screened }} 个
            </span>
            <button 
              @click="downloadCSV"
              class="px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-lg text-sm flex items-center gap-1"
            >
              📥 下载结果CSV
            </button>
          </div>
        </div>
        
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">smiles</th>
                <th v-if="drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100" @click="sortResults('bbbp_score')">
                  🧠 BBBP {{ sortColumn === 'bbbp_score' ? (sortAsc ? '↑' : '↓') : '' }}
                </th>
                <th v-if="drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100" @click="sortResults('esol_score')">
                  💧 ESOL {{ sortColumn === 'esol_score' ? (sortAsc ? '↑' : '↓') : '' }}
                </th>
                <th v-if="!drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100" @click="sortResults('score')">
                  score {{ sortColumn === 'score' ? (sortAsc ? '↑' : '↓') : '' }}
                </th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100" @click="sortResults('MolecularWeight')">
                  MolecularWeight {{ sortColumn === 'MolecularWeight' ? (sortAsc ? '↑' : '↓') : '' }}
                </th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">LogP</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">TPSA</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumHDonors</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumHAcceptors</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumAtoms</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumHeavyAtoms</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumRotatableBonds</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">NumAromaticRings</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr v-for="(mol, idx) in sortedResults" :key="idx" class="hover:bg-indigo-50 transition-colors">
                <td class="px-4 py-3 text-sm font-mono text-indigo-600 max-w-xs truncate" :title="mol.smiles">
                  {{ mol.smiles }}
                </td>
                <td v-if="drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-sm font-bold" :class="(mol.bbbp_score || 0) > 0.5 ? 'text-emerald-600' : 'text-amber-600'">
                  {{ mol.bbbp_score?.toFixed(4) || '-' }}
                </td>
                <td v-if="drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-sm font-bold" :class="(mol.esol_score || 0) > -3 ? 'text-blue-600' : 'text-amber-600'">
                  {{ mol.esol_score?.toFixed(4) || '-' }}
                </td>
                <td v-if="!drugStore.screeningResults.use_dual_model" class="px-4 py-3 text-sm font-bold text-indigo-600">{{ mol.score.toFixed(4) }}</td>
                <td class="px-4 py-3 text-sm" :class="(mol.properties.MolecularWeight || 0) > 500 ? 'text-red-500' : ''">
                  {{ mol.properties.MolecularWeight?.toFixed(3) || '-' }}
                </td>
                <td class="px-4 py-3 text-sm" :class="(mol.properties.LogP || 0) > 5 ? 'text-red-500' : ''">
                  {{ mol.properties.LogP?.toFixed(4) || '-' }}
                </td>
                <td class="px-4 py-3 text-sm">{{ mol.properties.TPSA?.toFixed(2) || '-' }}</td>
                <td class="px-4 py-3 text-sm" :class="(mol.properties.NumHDonors || 0) > 5 ? 'text-red-500' : ''">
                  {{ mol.properties.NumHDonors || '-' }}
                </td>
                <td class="px-4 py-3 text-sm" :class="(mol.properties.NumHAcceptors || 0) > 10 ? 'text-red-500' : ''">
                  {{ mol.properties.NumHAcceptors || '-' }}
                </td>
                <td class="px-4 py-3 text-sm">{{ mol.properties.NumAtoms || '-' }}</td>
                <td class="px-4 py-3 text-sm">{{ mol.properties.NumHeavyAtoms || '-' }}</td>
                <td class="px-4 py-3 text-sm">{{ mol.properties.NumRotatableBonds || '-' }}</td>
                <td class="px-4 py-3 text-sm">{{ mol.properties.NumAromaticRings || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Top Molecules Visualization -->
      <div class="bg-white shadow-lg rounded-xl p-6 border border-gray-100">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Top候选分子结构</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div v-for="(mol, idx) in drugStore.screeningResults.results.slice(0, 6)" :key="idx" 
               class="bg-gray-50 rounded-xl p-4 text-center">
            <img 
              :src="`http://localhost:8000/molecule/image?smiles=${encodeURIComponent(mol.smiles)}`"
              :alt="mol.smiles"
              class="w-full h-40 object-contain mb-3"
              @error="handleImageError"
            />
            <p class="text-sm font-medium text-gray-700">Rank {{ idx + 1 }}</p>
            <p class="text-sm text-indigo-600 font-mono truncate" :title="mol.smiles">Score: {{ mol.score.toFixed(4) }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useDrugStore } from '@/stores/drug';

const drugStore = useDrugStore();

// File upload
const fileInput = ref<HTMLInputElement | null>(null);
const uploadedFile = ref<File | null>(null);
const isDragging = ref(false);
const csvData = ref<any[]>([]);
const csvColumns = ref<string[]>([]);
const selectedSmilesColumn = ref('smiles');

// Manual input
const smilesText = ref('');

// Parameters
const topK = ref(10);
const ascending = ref(false);
const applyLipinski = ref(true);
const useDualModel = ref(true);

// Sorting
const sortColumn = ref('score');
const sortAsc = ref(false);

const triggerFileInput = () => {
  fileInput.value?.click();
};

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files[0]) {
    processFile(target.files[0]);
  }
};

const handleFileDrop = (event: DragEvent) => {
  isDragging.value = false;
  if (event.dataTransfer?.files && event.dataTransfer.files[0]) {
    processFile(event.dataTransfer.files[0]);
  }
};

const processFile = (file: File) => {
  if (!file.name.endsWith('.csv')) {
    alert('请上传CSV文件');
    return;
  }
  uploadedFile.value = file;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const text = e.target?.result as string;
    parseCSV(text);
  };
  reader.readAsText(file);
};

const parseCSV = (text: string) => {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return;
  
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  csvColumns.value = headers;
  
  // Auto-select smiles column
  const smilesCol = headers.find(h => h.toLowerCase().includes('smiles'));
  if (smilesCol) selectedSmilesColumn.value = smilesCol;
  
  csvData.value = lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
    const row: any = {};
    headers.forEach((h, i) => row[h] = values[i]);
    return row;
  }).filter(row => row[selectedSmilesColumn.value]);
};

const removeFile = () => {
  uploadedFile.value = null;
  csvData.value = [];
  csvColumns.value = [];
  if (fileInput.value) fileInput.value.value = '';
};

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
};

const handleScreen = () => {
  let smilesList: string[] = [];
  
  if (csvData.value.length > 0) {
    smilesList = csvData.value.map(row => row[selectedSmilesColumn.value]).filter(s => s);
  } else if (smilesText.value) {
    smilesList = smilesText.value.split('\n').map(s => s.trim()).filter(s => s.length > 0);
  }
  
  if (smilesList.length > 0) {
    drugStore.screenLibrary(smilesList, topK.value, applyLipinski.value, ascending.value, useDualModel.value);
  }
};

const handleImageError = (e: Event) => {
  (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="%23ccc">No Image</text></svg>';
};

const sortResults = (column: string) => {
  if (sortColumn.value === column) {
    sortAsc.value = !sortAsc.value;
  } else {
    sortColumn.value = column;
    sortAsc.value = true;
  }
};

const sortedResults = computed(() => {
  if (!drugStore.screeningResults?.results) return [];
  const results = [...drugStore.screeningResults.results];
  return results.sort((a, b) => {
    let aVal: number, bVal: number;
    if (sortColumn.value === 'score') {
      aVal = a.score;
      bVal = b.score;
    } else if (sortColumn.value === 'bbbp_score') {
      aVal = a.bbbp_score || 0;
      bVal = b.bbbp_score || 0;
    } else if (sortColumn.value === 'esol_score') {
      aVal = a.esol_score || 0;
      bVal = b.esol_score || 0;
    } else {
      aVal = a.properties[sortColumn.value] || 0;
      bVal = b.properties[sortColumn.value] || 0;
    }
    return sortAsc.value ? aVal - bVal : bVal - aVal;
  });
});

const downloadCSV = () => {
  if (!drugStore.screeningResults?.results) return;
  
  const headers = ['rank', 'smiles', 'score', 'MolecularWeight', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors'];
  const rows = drugStore.screeningResults.results.map((mol, idx) => [
    idx + 1,
    mol.smiles,
    mol.score.toFixed(4),
    mol.properties.MolecularWeight?.toFixed(2) || '',
    mol.properties.LogP?.toFixed(2) || '',
    mol.properties.TPSA?.toFixed(2) || '',
    mol.properties.NumHDonors || '',
    mol.properties.NumHAcceptors || ''
  ]);
  
  const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'screening_results.csv';
  link.click();
};
</script>