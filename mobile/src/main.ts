import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import ExploreView from './views/ExploreView.vue'
import './tokens.css'
import './base.css'
import './styles.css'
import './experience.css'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', component: ExploreView },
    { path: '/statistics', component: () => import('./views/CatalogStatisticsView.vue') },
    { path: '/atlas', component: () => import('./views/AtlasView.vue') },
    { path: '/atlas/compare', component: () => import('./views/AtlasCompareView.vue') },
    { path: '/atlas/public-imagery', redirect: '/atlas' },
    {
      path: '/atlas/type/:type',
      component: () => import('./views/AtlasTypeView.vue'),
      props: true,
    },
    { path: '/atlas/case/:id', component: () => import('./views/AtlasCaseView.vue'), props: true },
    {
      path: '/catalog/:id',
      name: 'catalog-record',
      component: () => import('./views/CatalogRecordView.vue'),
      props: true,
    },
    { path: '/recognize', component: () => import('./views/IdentifyView.vue') },
    { path: '/samples/:id', component: () => import('./views/SampleView.vue'), props: true },
    { path: '/chat', component: () => import('./views/ChatView.vue') },
    { path: '/mine', component: () => import('./views/MineView.vue') },
    { path: '/identify', redirect: '/recognize' },
    { path: '/assistant', redirect: '/chat' },
    { path: '/saved', redirect: '/mine' },
  ],
  scrollBehavior: (to) => (to.hash ? { el: to.hash, top: 16 } : { top: 0 }),
})
createApp(App).use(router).mount('#app')
