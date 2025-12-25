import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: '系统概览' }
  },
  {
    path: '/predict',
    name: 'Predict',
    component: () => import('@/views/PredictView.vue'),
    meta: { title: '单分子预测' }
  },
  {
    path: '/screen',
    name: 'Screen',
    component: () => import('@/views/ScreenView.vue'),
    meta: { title: '批量筛选' }
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// Update document title
router.beforeEach((to, _from, next) => {
  const title = to.meta.title as string;
  if (title) {
    document.title = `${title} - 药物筛选系统`;
  }
  next();
});

export default router;