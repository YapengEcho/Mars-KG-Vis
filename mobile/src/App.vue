<script setup lang="ts">
import { onMounted } from 'vue'
import { Bookmark, Compass, Layers, ScanLine, Sparkles } from '@lucide/vue'
import SlideDetail from './components/SlideDetail.vue'
import { atlas } from './stores/atlas'
const tabs = [
  { to: '/', label: '探索', icon: Compass },
  { to: '/atlas', label: '图鉴', icon: Layers },
  { to: '/recognize', label: '识别', icon: ScanLine },
  { to: '/chat', label: '助手', icon: Sparkles },
  { to: '/saved', label: '收藏', icon: Bookmark },
]
onMounted(() => atlas.load())
</script>
<template>
  <div class="app-shell">
    <header class="app-header">
      <RouterLink to="/" class="brand" aria-label="Mars Atlas 首页"
        ><span class="brand-mark"></span
        ><span
          >MARS<span class="brand-light"> ATLAS</span><small>火星滑坡探索</small></span
        ></RouterLink
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
