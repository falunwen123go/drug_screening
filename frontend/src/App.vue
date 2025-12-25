<template>
  <div class="min-h-screen flex flex-col bg-gray-50">
    <!-- Navbar -->
    <nav class="bg-indigo-600 text-white shadow-lg">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center">
            <router-link to="/" class="flex-shrink-0 font-bold text-xl">
              💊 药物筛选系统
            </router-link>
            <div class="ml-10 flex items-baseline space-x-4">
              <router-link 
                to="/" 
                class="px-3 py-2 rounded-md text-sm font-medium hover:bg-indigo-500 transition"
                active-class="bg-indigo-700"
              >
                系统概览
              </router-link>
              <router-link 
                to="/predict" 
                class="px-3 py-2 rounded-md text-sm font-medium hover:bg-indigo-500 transition"
                active-class="bg-indigo-700"
              >
                单分子预测
              </router-link>
              <router-link 
                to="/screen" 
                class="px-3 py-2 rounded-md text-sm font-medium hover:bg-indigo-500 transition"
                active-class="bg-indigo-700"
              >
                批量筛选
              </router-link>
            </div>
          </div>
          <div>
            <span class="text-xs bg-indigo-800 px-2 py-1 rounded text-indigo-200">v1.0.0</span>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="flex-grow max-w-7xl w-full mx-auto py-6 sm:px-6 lg:px-8">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200">
      <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 flex flex-col items-center space-y-2">
        <p class="text-center text-gray-500 text-sm">
          &copy; 2025 智能药物筛选系统. Powered by PyTorch & FastAPI.
        </p>
        <a 
          href="https://github.com/falunwen123go/drug_screening" 
          target="_blank" 
          rel="noopener noreferrer"
          class="text-gray-400 hover:text-indigo-600 transition flex items-center text-sm"
        >
          <svg class="h-5 w-5 mr-1" fill="currentColor" viewBox="0 0 24 24">
            <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.987 1.029-2.683-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.696 1.028 1.59 1.028 2.683 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd" />
          </svg>
          GitHub 仓库
        </a>
      </div>
    </footer>
  </div>
</template>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>