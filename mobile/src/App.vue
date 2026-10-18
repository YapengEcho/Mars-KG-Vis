<script setup lang="ts">
import { onMounted } from 'vue'
import { Bookmark, Compass, Layers, ScanLine } from '@lucide/vue'
import SlideDetail from './components/SlideDetail.vue'
import { atlas } from './stores/atlas'
const tabs = [
  { to: '/', label: '探索', icon: Compass },
  { to: '/atlas', label: '图鉴', icon: Layers },
  { to: '/assistant', label: '助手', icon: ScanLine },
  { to: '/saved', label: '收藏', icon: Bookmark },
]
onMounted(() => atlas.load())
</script>
<template>
  <div class="app-shell">
    <header class="app-header">
      <RouterLink to="/" class="brand" aria-label="Mars Atlas 首页"
        ><span class="brand-orbit"></span
        ><span
          >MARS<span class="brand-light"> ATLAS</span><small>火星滑坡探索</small></span
        ></RouterLink
      ><span class="mode-pill"
        ><span></span
        >{{
          atlas.state.loading
            ? '加载中'
            : atlas.state.catalog.mode === 'demo'
              ? '示例目录'
              : '滑坡目录'
        }}</span
      >
    </header>
    <div v-if="atlas.state.error || atlas.state.storageError" class="notice" role="status">
      {{ atlas.state.error || atlas.state.storageError }}
    </div>
    <main id="main-content"><RouterView /></main>
    <nav class="bottom-nav" aria-label="主导航">
      <RouterLink v-for="tab in tabs" :key="tab.to" :to="tab.to" :exact-active-class="'is-active'"
        ><component :is="tab.icon" :size="21" :stroke-width="1.6" /><span>{{ tab.label }}</span
        ><i></i
      ></RouterLink>
    </nav>
    <SlideDetail />
  </div>
</template>
