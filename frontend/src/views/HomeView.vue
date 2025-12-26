<template>
  <div class="space-y-8">
    <!-- Welcome Banner -->
    <div class="bg-gradient-to-r from-indigo-600 to-blue-500 rounded-lg shadow-lg p-8 text-white">
      <h1 class="text-3xl font-bold mb-2">欢迎使用 AI 药物筛选平台</h1>
      <p class="opacity-90">
        基于深度学习的高级分子属性预测与虚拟筛选系统。实时监控系统状态，动态切换预测模型。
      </p>
    </div>

    <!-- Screening Funnel Diagram -->
    <ScreeningFunnel />

    <div v-if="systemStore.loading && !systemStore.info" class="text-center py-12">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
        <p class="mt-4 text-gray-500">正在获取系统信息...</p>
    </div>

    <div v-else-if="systemStore.error" class="bg-red-50 border-l-4 border-red-500 p-4 rounded shadow">
        <div class="flex">
            <div class="flex-shrink-0">⚠️</div>
            <div class="ml-3">
                <p class="text-sm text-red-700">{{ systemStore.error }}</p>
                <button @click="systemStore.fetchSystemInfo()" class="mt-2 text-red-700 underline text-sm hover:text-red-900">重试</button>
            </div>
        </div>
    </div>

    <div v-else class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      <!-- Left Column: System Status (2/3 width) -->
      <div class="lg:col-span-2 space-y-6">
        <h2 class="text-xl font-bold text-gray-800 flex items-center">
            <span class="mr-2">📊</span> 系统状态监控
        </h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Service Status -->
            <div class="bg-white p-5 rounded-lg shadow border-t-4" :class="isConnected ? 'border-green-500' : 'border-red-500'">
                <h3 class="text-sm font-medium text-gray-500 uppercase">服务状态</h3>
                <div class="mt-2 flex items-center justify-between">
                    <span class="text-2xl font-bold" :class="isConnected ? 'text-green-600' : 'text-red-600'">
                        {{ isConnected ? '运行中' : '异常' }}
                    </span>
                    <span class="h-3 w-3 rounded-full" :class="isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'"></span>
                </div>
                <p class="text-xs text-gray-400 mt-2">API 响应正常</p>
            </div>

            <!-- Compute Device -->
            <div class="bg-white p-5 rounded-lg shadow border-t-4 border-blue-500">
                <h3 class="text-sm font-medium text-gray-500 uppercase">计算设备</h3>
                <div class="mt-2">
                    <span class="text-2xl font-bold text-gray-800">
                        {{ systemStore.info?.device === 'cuda' ? 'GPU' : 'CPU' }}
                    </span>
                </div>
                <p class="text-xs text-gray-400 mt-2" v-if="systemStore.info?.gpu_info">
                    {{ systemStore.info.gpu_info.name }} (x{{ systemStore.info.gpu_info.count }})
                </p>
                <p class="text-xs text-gray-400 mt-2" v-else>
                    Standard Processor
                </p>
            </div>
        </div>

        <!-- Detailed Specs -->
        <div class="bg-white rounded-lg shadow overflow-hidden">
            <div class="px-6 py-4 border-b border-gray-200 bg-gray-50">
                <h3 class="text-lg font-medium text-gray-900">硬件详情</h3>
            </div>
            <div class="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- CPU Info -->
                <div>
                    <h4 class="font-semibold text-gray-700 mb-2">CPU 信息</h4>
                    <ul class="space-y-2 text-sm text-gray-600">
                        <li class="flex justify-between">
                            <span>架构:</span>
                            <span class="font-mono text-gray-900">{{ systemStore.info?.cpu_info.machine }}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>处理器:</span>
                            <span class="font-mono text-gray-900 truncate max-w-[150px]" :title="systemStore.info?.cpu_info.processor">{{ systemStore.info?.cpu_info.processor }}</span>
                        </li>
                        <li class="flex justify-between">
                            <span>系统:</span>
                            <span class="font-mono text-gray-900">{{ systemStore.info?.cpu_info.system }}</span>
                        </li>
                    </ul>
                </div>

                <!-- Memory Info -->
                <div v-if="systemStore.info?.memory_info && Object.keys(systemStore.info.memory_info).length > 0">
                    <h4 class="font-semibold text-gray-700 mb-2">内存使用</h4>
                    <div class="relative pt-1">
                        <div class="overflow-hidden h-2 mb-4 text-xs flex rounded bg-indigo-200">
                            <div :style="`width: ${systemStore.info.memory_info.percent}`" class="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-indigo-500 transition-all duration-500"></div>
                        </div>
                        <ul class="space-y-2 text-sm text-gray-600">
                            <li class="flex justify-between">
                                <span>利用率:</span>
                                <span class="font-bold text-indigo-600">{{ systemStore.info.memory_info.percent }}</span>
                            </li>
                            <li class="flex justify-between">
                                <span>总内存:</span>
                                <span class="font-mono text-gray-900">{{ systemStore.info.memory_info.total }}</span>
                            </li>
                            <li class="flex justify-between">
                                <span>可用:</span>
                                <span class="font-mono text-gray-900">{{ systemStore.info.memory_info.available }}</span>
                            </li>
                        </ul>
                    </div>
                </div>
                <div v-else class="flex items-center justify-center text-gray-400 italic text-sm">
                    内存信息不可用 (psutil 未安装)
                </div>
            </div>
        </div>
      </div>

      <!-- Right Column: Model Control (1/3 width) -->
      <div class="space-y-6">
        <h2 class="text-xl font-bold text-gray-800 flex items-center">
            <span class="mr-2">🧠</span> 模型管理
        </h2>
        
        <div class="bg-white rounded-lg shadow p-6">
            <div class="mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-2">当前加载模型</label>
                <div class="flex items-center p-3 bg-indigo-50 rounded-md border border-indigo-100">
                    <span class="text-xl mr-3">📦</span>
                    <span class="font-mono text-indigo-800 font-semibold truncate" :title="systemStore.info?.current_model || '无'">
                        {{ systemStore.info?.current_model || '未加载模型' }}
                    </span>
                </div>
            </div>

            <div>
                <label for="model-select" class="block text-sm font-medium text-gray-700 mb-2">切换模型</label>
                <select 
                    id="model-select" 
                    v-model="selectedModel" 
                    class="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                >
                    <option disabled value="">请选择模型...</option>
                    <option v-for="model in systemStore.info?.available_models" :key="model" :value="model">
                        {{ model }}
                    </option>
                </select>
                
                <button 
                    @click="handleLoadModel" 
                    :disabled="!selectedModel || selectedModel === systemStore.info?.current_model || systemStore.loadingModel"
                    class="mt-4 w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                    <span v-if="systemStore.loadingModel" class="animate-spin mr-2">↻</span>
                    {{ systemStore.loadingModel ? '加载中...' : '加载选中模型' }}
                </button>
            </div>
            
            <p class="mt-4 text-xs text-gray-500">
                注意：切换模型可能需要几秒钟重新初始化权重。请确保所选模型与当前的预测任务类型兼容。
            </p>
        </div>

        <!-- Quick Actions (Optional) -->
        <div class="bg-white rounded-lg shadow p-6">
             <h3 class="text-sm font-medium text-gray-500 uppercase mb-4">快捷操作</h3>
             <div class="space-y-3">
                 <button @click="systemStore.fetchSystemInfo()" class="w-full flex items-center justify-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                     🔄 刷新系统状态
                 </button>
             </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue';
import { useSystemStore } from '@/stores/system';
import ScreeningFunnel from '@/components/ScreeningFunnel.vue';

const systemStore = useSystemStore();
const selectedModel = ref('');

const isConnected = computed(() => systemStore.info?.status === 'healthy' || systemStore.info?.status === 'no_model');

// Update selected model when current model changes (initial load)
watch(() => systemStore.info?.current_model, (newVal) => {
    if (newVal) selectedModel.value = newVal;
}, { immediate: true });

const handleLoadModel = async () => {
    if (selectedModel.value) {
        await systemStore.loadModel(selectedModel.value);
    }
};

onMounted(() => {
  systemStore.fetchSystemInfo();
});
</script>