# AI 药物筛选系统 - 前端 (Frontend)

基于 Vue 3 + TypeScript + Vite 构建的药物筛选系统前端界面。提供直观的用户交互，用于药物分子的属性预测、批量筛选以及系统状态监控。

## 🛠️ 技术栈

- **核心框架:** [Vue 3](https://vuejs.org/) (Script Setup, Composition API)
- **构建工具:** [Vite](https://vitejs.dev/)
- **语言:** [TypeScript](https://www.typescriptlang.org/)
- **状态管理:** [Pinia](https://pinia.vuejs.org/)
- **路由管理:** [Vue Router](https://router.vuejs.org/)
- **HTTP 客户端:** [Axios](https://axios-http.com/)
- **样式框架:** [Tailwind CSS](https://tailwindcss.com/)

## ✨ 功能特性

1.  **系统概览 (Home):**
    - 实时监控后端服务状态 (CPU/GPU/内存)。
    - 动态切换预测模型 (如 BBBP, ESOL 等)。
    - 查看硬件详细信息。

2.  **单分子预测 (Single Prediction):**
    - 输入 SMILES 字符串进行实时预测。
    - 可视化显示预测概率和 Lipinski 五规则符合情况。
    - 展示关键物理化学属性 (MW, LogP, TPSA, H-Donors, H-Acceptors)。

3.  **批量筛选 (Batch Screening):**
    - 支持批量输入 SMILES 列表。
    - 可配置 Top-K 筛选和 Lipinski 规则过滤。
    - 表格化展示筛选结果，支持按评分排序。

## 🚀 快速开始

### 前置要求

- Node.js (推荐 v18+ )
- pnpm (推荐) 或 npm/yarn

### 1. 安装依赖

```bash
cd frontend
pnpm install
# 或者
npm install
```

### 2. 开发环境运行

启动开发服务器，默认端口通常为 5173。
**注意：** 确保后端服务已在 `http://127.0.0.1:8000` 启动，否则 API 请求将失败。

```bash
pnpm dev
# 或者
npm run dev
```

### 3. 生产环境构建

构建用于生产环境的静态文件，输出目录为 `dist`。

```bash
pnpm build
# 或者
npm run build
```

### 4. 预览构建结果

```bash
pnpm preview
# 或者
npm run preview
```

## 📂 目录结构

```
frontend/
├── src/
│   ├── api/            # Axios 封装及 API 请求
│   ├── assets/         # 静态资源
│   ├── components/     # 公共组件
│   ├── layout/         # 布局组件
│   ├── router/         # 路由配置
│   ├── stores/         # Pinia 状态管理 (Drug, System)
│   ├── types/          # TypeScript 类型定义
│   ├── views/          # 页面视图 (Home, Predict, Screen)
│   ├── App.vue         # 根组件
│   └── main.ts         # 入口文件
├── index.html
├── package.json
├── tailwind.config.js  # Tailwind 配置
├── tsconfig.json       # TypeScript 配置
└── vite.config.ts      # Vite 配置 (包含 API 代理)
```

## ⚙️ 配置说明

### API 代理

在 `vite.config.ts` 中配置了开发环境的反向代理，将 `/api` 开头的请求转发至后端：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

如需修改后端地址，请调整 `target` 字段。