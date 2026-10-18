<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { Search, ArrowRight, LocateFixed, Minus, Plus, X } from '@lucide/vue'
import { useRoute } from 'vue-router'
import MarsGlobe from '../components/MarsGlobe.vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas } from '../stores/atlas'
import { filterSlides } from '../lib/catalog'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'
const query = ref('')
const type = ref<SlideType | ''>('')
const route = useRoute()
const globe = ref<InstanceType<typeof MarsGlobe>>()
const focusedId = ref('')
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
async function focusFromRoute() {
  const id = route.query.focus
  if (typeof id !== 'string') return
  query.value = ''
  type.value = ''
  focusedId.value = id
  await nextTick()
  globe.value?.focusPoint(id)
}
watch(() => route.fullPath, focusFromRoute)
onMounted(focusFromRoute)
function select(id: string) {
  focusedId.value = id
  atlas.select(id)
}
function reset() {
  focusedId.value = ''
  globe.value?.resetView()
}
</script>
<template>
  <section class="explore-view">
    <div class="explore-heading">
      <div>
        <span class="eyebrow">MARS LANDSLIDES</span>
        <h1>探索火星滑坡</h1>
        <p>浏览滑坡分布、类型与案例。</p>
      </div>
    </div>
    <div class="search-box">
      <Search :size="19" /><input
        v-model="query"
        aria-label="搜索滑坡编号或名称"
        placeholder="搜索编号、名称或类型"
        inputmode="search"
      /><button v-if="query" class="icon-button" aria-label="清除搜索" @click="query = ''">
        <X :size="16" />
      </button>
    </div>
    <div class="type-filters" role="group" aria-label="类型筛选">
      <button :class="{ active: !type }" :aria-pressed="!type" @click="type = ''">全部</button
      ><button
        v-for="t in slideTypes"
        :key="t"
        :class="{ active: type === t }"
        :aria-pressed="type === t"
        @click="type = type === t ? '' : t"
      >
        <i :style="{ background: typeInfo[t].color }"></i>{{ typeInfo[t].name }}
      </button>
    </div>
    <div class="globe-stage">
      <div class="globe-topline">
        <span><i></i>MARS · {{ type || 'ALL TYPES' }}</span
        ><span>三维球体示意</span>
      </div>
      <MarsGlobe ref="globe" :points="matches" :selected-id="focusedId" @select="select" />
      <div class="globe-tools">
        <button class="icon-button" aria-label="放大火星球" @click="globe?.zoomIn()">
          <Plus :size="20" /></button
        ><button class="icon-button" aria-label="缩小火星球" @click="globe?.zoomOut()">
          <Minus :size="20" /></button
        ><button class="icon-button" aria-label="复位火星球" @click="reset">
          <LocateFixed :size="19" />
        </button>
      </div>
      <div class="globe-footer">
        <span>拖动旋转 · 双指缩放 · 点选查看</span><span>纹理为视觉示意</span>
      </div>
    </div>
    <section class="discovery-section">
      <div class="section-heading">
        <div>
          <span class="eyebrow">LANDSLIDE CATALOG</span>
          <h2>
            {{ query || type ? '查询结果' : '滑坡目录' }} <span>{{ matches.length }}</span>
          </h2>
        </div>
        <RouterLink to="/atlas">查看目录 <ArrowRight :size="16" /></RouterLink>
      </div>
      <div v-if="matches.length" class="discovery-cards">
        <SlideCard v-for="slide in matches.slice(0, 6)" :key="slide.id" :slide="slide" compact />
      </div>
      <div v-else class="empty-inline">没有匹配的点位，试试其他编号或类型。</div>
    </section>
    <p v-if="atlas.state.catalog.mode === 'demo'" class="page-footnote">
      示例目录用于体验浏览流程，不代表真实滑坡位置或分类。
    </p>
  </section>
</template>
