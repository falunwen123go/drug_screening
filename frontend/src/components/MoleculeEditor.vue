<template>
  <div class="molecule-editor">
    <div class="flex items-center justify-between mb-3">
      <h3 class="text-sm font-medium text-gray-700">🎨 分子编辑器 (JSME)</h3>
      <div class="flex gap-2">
        <button 
          @click="clearEditor"
          class="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded text-gray-600"
        >
          清空
        </button>
        <button 
          @click="getSmilesFromEditor"
          class="px-3 py-1 text-xs bg-emerald-500 hover:bg-emerald-600 text-white rounded font-medium"
        >
          获取SMILES
        </button>
      </div>
    </div>
    
    <!-- JSME Editor Container -->
    <div 
      ref="editorContainer" 
      id="jsme-container"
      class="border rounded-lg overflow-hidden bg-white flex items-center justify-center"
      :style="{ height: height + 'px' }"
    >
      <div v-if="!editorLoaded" class="text-gray-400">
        <div class="animate-pulse">⏳ 加载编辑器中...</div>
      </div>
    </div>
    
    <!-- SMILES Output -->
    <div v-if="currentSmiles" class="mt-3 p-3 bg-gray-50 rounded-lg">
      <div class="flex items-center justify-between">
        <span class="text-xs text-gray-500">生成的SMILES:</span>
        <button 
          @click="copySmiles"
          class="text-xs text-indigo-600 hover:text-indigo-700"
        >
          📋 复制
        </button>
      </div>
      <code class="block mt-1 text-sm font-mono text-indigo-600 break-all">{{ currentSmiles }}</code>
    </div>
    
    <!-- Help Info -->
    <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
      <p class="text-xs text-blue-700">
        💡 <strong>使用提示：</strong>
      </p>
      <ul class="mt-1 text-xs text-blue-600 list-disc list-inside space-y-1">
        <li>点击左侧工具栏选择原子（C、N、O等）和键类型</li>
        <li>在画布中绘制分子结构</li>
        <li>绘制完成后点击"获取SMILES"按钮</li>
        <li>或者直接在上方"手动输入"标签页输入SMILES</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';

const props = defineProps<{
  height?: number;
  initialSmiles?: string;
}>();

const emit = defineEmits<{
  (e: 'update:smiles', smiles: string): void;
}>();

const editorContainer = ref<HTMLDivElement | null>(null);
const currentSmiles = ref('');
const editorLoaded = ref(false);
let jsmeApplet: any = null;

onMounted(() => {
  loadJSME();
});

const loadJSME = () => {
  // 加载 JSME 库
  const script = document.createElement('script');
  script.src = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js';
  script.onload = () => {
    initJSME();
  };
  script.onerror = () => {
    console.error('Failed to load JSME editor');
    editorLoaded.value = false;
  };
  document.head.appendChild(script);
};

const initJSME = () => {
  setTimeout(() => {
    try {
      // @ts-ignore
      if (typeof JSApplet !== 'undefined') {
        // @ts-ignore
        jsmeApplet = new JSApplet.JSME('jsme-container', `${props.height || 400}px`, '100%', {
          options: 'query,hydrogens'
        });
        editorLoaded.value = true;
        
        // 如果有初始SMILES，设置到编辑器
        if (props.initialSmiles) {
          setSmilesToEditor(props.initialSmiles);
        }
      }
    } catch (e) {
      console.error('Failed to initialize JSME:', e);
    }
  }, 500);
};

const getSmilesFromEditor = () => {
  if (jsmeApplet) {
    try {
      const smiles = jsmeApplet.smiles();
      currentSmiles.value = smiles;
      emit('update:smiles', smiles);
    } catch (e) {
      console.error('Failed to get SMILES:', e);
    }
  }
};

const setSmilesToEditor = (smiles: string) => {
  if (jsmeApplet && smiles) {
    try {
      jsmeApplet.readGenericMolecularInput(smiles);
    } catch (e) {
      console.error('Failed to set SMILES:', e);
    }
  }
};

const clearEditor = () => {
  if (jsmeApplet) {
    try {
      jsmeApplet.clear();
      currentSmiles.value = '';
      emit('update:smiles', '');
    } catch (e) {
      console.error('Failed to clear editor:', e);
    }
  }
};

const copySmiles = () => {
  if (currentSmiles.value) {
    navigator.clipboard.writeText(currentSmiles.value);
  }
};

// 监听外部SMILES变化
watch(() => props.initialSmiles, (newSmiles) => {
  if (newSmiles && editorLoaded.value) {
    setSmilesToEditor(newSmiles);
  }
});

defineExpose({
  getSmiles: getSmilesFromEditor,
  setSmiles: setSmilesToEditor,
  clear: clearEditor
});
</script>

<style scoped>
.molecule-editor {
  width: 100%;
}
</style>
