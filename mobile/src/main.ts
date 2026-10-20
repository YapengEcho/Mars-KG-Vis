import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import ExploreView from './views/ExploreView.vue'
import './styles.css'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', component: ExploreView },
    { path: '/atlas', component: () => import('./views/AtlasView.vue') },
    { path: '/identify', component: () => import('./views/IdentifyView.vue') },
    { path: '/assistant', component: () => import('./views/AssistantView.vue') },
    { path: '/mine', component: () => import('./views/MineView.vue') },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
createApp(App).use(router).mount('#app')
