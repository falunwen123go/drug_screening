<template>
  <div class="screening-funnel bg-white rounded-2xl shadow-xl p-8 overflow-hidden">
    <!-- Title -->
    <div class="text-center mb-8">
      <h2 class="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">
        🔬 "数字漏斗"：虚拟筛选的核心逻辑
      </h2>
      <p class="text-gray-500 mt-2 text-sm">
        从海量候选化合物中，层层筛选出最具潜力的药物分子
      </p>
    </div>

    <!-- Funnel Visualization -->
    <div class="relative">
      <!-- Background Funnel Shape -->
      <div class="funnel-bg absolute inset-0 opacity-5">
        <svg viewBox="0 0 400 500" class="w-full h-full">
          <path d="M50,0 L350,0 L300,150 L280,250 L260,350 L220,500 L180,500 L140,350 L120,250 L100,150 Z" 
                fill="url(#funnelGradient)"/>
          <defs>
            <linearGradient id="funnelGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:#6366f1"/>
              <stop offset="100%" style="stop-color:#10b981"/>
            </linearGradient>
          </defs>
        </svg>
      </div>

      <!-- Steps Container -->
      <div class="relative z-10 space-y-4">
        
        <!-- Step 1: Input -->
        <div class="funnel-step" :class="{ 'active': activeStep === 1 }" @mouseenter="activeStep = 1">
          <div class="step-content w-full bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-t-2xl p-6 text-white shadow-lg transform hover:scale-[1.02] transition-all">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <div class="step-number bg-white/20 rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold">
                  1
                </div>
                <div>
                  <h3 class="font-bold text-lg">📥 输入海量候选分子</h3>
                  <p class="text-indigo-100 text-sm mt-1">上传包含 SMILES 的化合物库文件</p>
                </div>
              </div>
              <div class="text-right">
                <div class="text-xs text-indigo-200">候选化合物</div>
              </div>
            </div>
            <div class="mt-4 flex flex-wrap gap-2">
              <span class="px-2 py-1 bg-white/10 rounded text-xs">CSV文件上传</span>
              <span class="px-2 py-1 bg-white/10 rounded text-xs">手动输入</span>
              <span class="px-2 py-1 bg-white/10 rounded text-xs">数据库导入</span>
            </div>
          </div>
        </div>

        <!-- Arrow 1 -->
        <div class="flex justify-center -my-2 relative z-20">
          <div class="arrow-down">
            <svg class="w-8 h-8 text-indigo-400 animate-bounce" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
          </div>
        </div>

        <!-- Step 2: ESOL Filter -->
        <div class="funnel-step mx-4" :class="{ 'active': activeStep === 2 }" @mouseenter="activeStep = 2">
          <div class="step-content bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl p-5 text-white shadow-lg transform hover:scale-[1.02] transition-all">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <div class="step-number bg-white/20 rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">
                  2
                </div>
                <div>
                  <h3 class="font-bold">💧 水溶性初筛 (ESOL模型)</h3>
                  <p class="text-blue-100 text-sm mt-1">预测水溶解度，剔除难溶解分子</p>
                </div>
              </div>
              <div class="text-right">
                <div class="flex items-center gap-2">

                </div>

              </div>
            </div>
            <div class="mt-3 bg-white/10 rounded-lg p-3">
              <div class="flex items-center justify-between text-sm">
                <span>筛选条件：</span>
                <code class="bg-white/10 px-2 py-0.5 rounded">ESOL &gt; -3.0 log mol/L</code>
              </div>
            </div>
          </div>
        </div>

        <!-- Arrow 2 -->
        <div class="flex justify-center -my-2 relative z-20">
          <div class="arrow-down">
            <svg class="w-8 h-8 text-blue-400 animate-bounce" style="animation-delay: 0.1s" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
          </div>
        </div>

        <!-- Step 3: BBBP Filter -->
        <div class="funnel-step mx-8" :class="{ 'active': activeStep === 3 }" @mouseenter="activeStep = 3">
          <div class="step-content bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl p-5 text-white shadow-lg transform hover:scale-[1.02] transition-all">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <div class="step-number bg-white/20 rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">
                  3
                </div>
                <div>
                  <h3 class="font-bold">🧠 血脑屏障筛选 (BBBP模型)</h3>
                  <p class="text-purple-100 text-sm mt-1">预测穿透BBB能力（CNS药物适用）</p>
                </div>
              </div>
              <div class="text-right">
                <div class="flex items-center gap-2">

                </div>

              </div>
            </div>
            <div class="mt-3 bg-white/10 rounded-lg p-3">
              <div class="flex items-center justify-between text-sm">
                <span>筛选条件：</span>
                <code class="bg-white/10 px-2 py-0.5 rounded">BBBP &gt; 0.5 (高概率穿透)</code>
              </div>
            </div>
          </div>
        </div>

        <!-- Arrow 3 -->
        <div class="flex justify-center -my-2 relative z-20">
          <div class="arrow-down">
            <svg class="w-8 h-8 text-purple-400 animate-bounce" style="animation-delay: 0.2s" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
          </div>
        </div>

        <!-- Step 4: Lipinski Filter -->
        <div class="funnel-step mx-12" :class="{ 'active': activeStep === 4 }" @mouseenter="activeStep = 4">
          <div class="step-content bg-gradient-to-r from-amber-500 to-orange-500 rounded-xl p-5 text-white shadow-lg transform hover:scale-[1.02] transition-all">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <div class="step-number bg-white/20 rounded-full w-10 h-10 flex items-center justify-center text-lg font-bold">
                  4
                </div>
                <div>
                  <h3 class="font-bold">💊 类药性规则过滤 (Lipinski)</h3>
                  <p class="text-amber-100 text-sm mt-1">检查分子的成药潜力</p>
                </div>
              </div>
              <div class="text-right">
                <div class="flex items-center gap-2">

                </div>

              </div>
            </div>
            <div class="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div class="bg-white/10 rounded px-2 py-1">MW ≤ 500 Da</div>
              <div class="bg-white/10 rounded px-2 py-1">LogP ≤ 5</div>
              <div class="bg-white/10 rounded px-2 py-1">HBD ≤ 5</div>
              <div class="bg-white/10 rounded px-2 py-1">HBA ≤ 10</div>
            </div>
          </div>
        </div>

        <!-- Arrow 4 -->
        <div class="flex justify-center -my-2 relative z-20">
          <div class="arrow-down">
            <svg class="w-8 h-8 text-amber-400 animate-bounce" style="animation-delay: 0.3s" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
          </div>
        </div>

        <!-- Step 5: Output -->
        <div class="funnel-step mx-16" :class="{ 'active': activeStep === 5 }" @mouseenter="activeStep = 5">
          <div class="step-content bg-gradient-to-r from-emerald-500 to-green-500 rounded-b-2xl p-6 text-white shadow-lg transform hover:scale-[1.02] transition-all border-4 border-emerald-300">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-4">
                <div class="step-number bg-white/30 rounded-full w-12 h-12 flex items-center justify-center text-xl font-bold">
                  ✓
                </div>
                <div>
                  <h3 class="font-bold text-lg">🏆 最终产出：高潜力候选化合物</h3>
                  <p class="text-emerald-100 text-sm mt-1">导出 Top-K 最优分子进行实验验证</p>
                </div>
              </div>
              <div class="text-right">
                <div class="text-3xl font-bold">Top 10</div>
                <div class="text-xs text-emerald-200">精选候选药物</div>
              </div>
            </div>
            <div class="mt-4 flex items-center gap-3">
              <span class="px-3 py-1 bg-white/20 rounded-full text-sm">📊 分数排名</span>
              <span class="px-3 py-1 bg-white/20 rounded-full text-sm">📥 CSV导出</span>
              <span class="px-3 py-1 bg-white/20 rounded-full text-sm">🔬 可视化</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Statistics Bar -->
    <div class="mt-8 grid grid-cols-4 gap-4">
      
    </div>

    <!-- Legend -->
    <div class="mt-6 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">📋 筛选原理说明</h4>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
        <div class="flex items-start gap-2">
          <span class="text-blue-500">💧</span>
          <div>
            <strong>ESOL模型</strong>：预测分子在水中的溶解度(log mol/L)，
            溶解性差的分子难以被吸收，优先剔除。
          </div>
        </div>
        <div class="flex items-start gap-2">
          <span class="text-purple-500">🧠</span>
          <div>
            <strong>BBBP模型</strong>：预测分子穿透血脑屏障的概率，
            针对CNS药物（如抗抑郁药）尤为重要。
          </div>
        </div>
        <div class="flex items-start gap-2">
          <span class="text-amber-500">💊</span>
          <div>
            <strong>Lipinski五规则</strong>：经典的类药性评估标准，
            符合规则的分子更可能成为口服药物。
          </div>
        </div>
        <div class="flex items-start gap-2">
          <span class="text-emerald-500">🏆</span>
          <div>
            <strong>Top-K排序</strong>：按预测分数降序排列，
            选取最优的K个候选进入后续实验验证。
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const activeStep = ref(0);
</script>

<style scoped>
.screening-funnel {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

.funnel-step {
  transition: all 0.3s ease;
}

.funnel-step.active .step-content {
  box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.2);
}

.step-number {
  backdrop-filter: blur(4px);
}

.arrow-down {
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.funnel-step:hover .step-content {
  animation: float 2s ease-in-out infinite;
}
</style>
