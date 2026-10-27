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
    { path: '/recognize', component: () => import('./views/IdentifyView.vue') },
    { path: '/chat', component: () => import('./views/ChatView.vue') },
    { path: '/mine', component: () => import('./views/MineView.vue') },
    { path: '/identify', redirect: '/recognize' },
    { path: '/assistant', redirect: '/chat' },
    { path: '/saved', redirect: '/mine' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
createApp(App).use(router).mount('#app')
