<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { Search, ArrowRight, LocateFixed, Minus, Plus, X } from '@lucide/vue'
import { useRoute } from 'vue-router'
import MarsGlobe from '../components/MarsGlobe.vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas } from '../stores/atlas'
import { filterSlides, findSlide } from '../lib/catalog'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'
const pageSize = 20
const query = ref('')
const type = ref<SlideType | ''>('')
const route = useRoute()
const globe = ref<InstanceType<typeof MarsGlobe>>()
const focusedId = ref('')
const focusNote = ref('')
const visibleCount = ref(pageSize)
const basemap = ref<'loading' | 'ready' | 'fallback'>('loading')
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
const visible = computed(() => matches.value.slice(0, visibleCount.value))
const remaining = computed(() => Math.max(0, matches.value.length - visible.value.length))
const modeLabel = computed(() =>
  atlas.state.catalog.mode === 'catalog'
    ? `真实目录 · ${atlas.state.catalog.version}`
    : `示例数据 · ${atlas.state.catalog.version}`,
)
const basemapNote = computed(
  () =>
    ({
      loading: '正在加载全球底图…',
      ready: '底图：Solar System Scope（CC BY 4.0）',
      fallback: '底图未加载 · 点位按经纬度显示',
    })[basemap.value],
)
watch([query, type], () => {
  visibleCount.value = pageSize
})
async function focusFromRoute() {
  const key = route.query.focus
  if (typeof key !== 'string' || !key.trim()) return
  const record = findSlide(atlas.state.catalog.records, key)
  if (!record) {
    focusNote.value = atlas.state.loading
      ? `正在加载目录，稍后定位编号 ${key}`
      : `未找到编号 ${key}，请确认目录来源`
    return
  }
  query.value = ''
  type.value = ''
  focusedId.value = record.id
  focusNote.value = `已定位 ${record.sourceId ?? record.id}`
  await nextTick()
  globe.value?.focusPoint(record.id)
  const index = matches.value.findIndex((item) => item.id === record.id)
  if (index >= visibleCount.value) visibleCount.value = index + 1
}
watch(() => route.fullPath, focusFromRoute)
watch(() => [atlas.state.loading, atlas.state.catalog.version], focusFromRoute)
onMounted(focusFromRoute)
function select(id: string) {
  focusedId.value = id
  atlas.select(id)
}
function reset() {
  focusedId.value = ''
  globe.value?.resetView()
}
function loadMore() {
  visibleCount.value += pageSize
}
function clearFilters() {
  query.value = ''
  type.value = ''
  focusNote.value = ''
}
function clearQuery() {
  query.value = ''
  focusNote.value = ''
}
function toggleType(value: SlideType) {
  type.value = type.value === value ? '' : value
  focusNote.value = ''
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
      <span class="data-badge" :class="atlas.state.catalog.mode" aria-live="polite">{{
        atlas.state.loading ? '目录加载中…' : modeLabel
      }}</span>
    </div>
    <div class="search-box">
      <Search :size="19" /><input
        v-model="query"
        aria-label="搜索滑坡编号、名称或说明"
        placeholder="搜索编号、名称或说明"
        inputmode="search"
        @input="focusNote = ''"
      /><button v-if="query" class="icon-button" aria-label="清除搜索" @click="clearQuery">
        <X :size="16" />
      </button>
    </div>
    <div class="type-filters" role="group" aria-label="类型筛选">
      <button :class="{ active: !type }" :aria-pressed="!type" @click="clearFilters()">全部</button
      ><button
        v-for="t in slideTypes"
        :key="t"
        :class="{ active: type === t }"
        :aria-pressed="type === t"
        @click="toggleType(t)"
      >
        <i :style="{ background: typeInfo[t].color }"></i>{{ typeInfo[t].name }}
      </button>
    </div>
    <div class="globe-stage">
      <div class="globe-topline">
        <span><i></i>MARS · {{ type || 'ALL TYPES' }}</span
        ><span>{{ matches.length }} 个点位</span>
      </div>
      <MarsGlobe
        ref="globe"
        :points="matches"
        :selected-id="focusedId"
        @select="select"
        @basemap="basemap = $event"
      />
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
        <span>拖动旋转 · 双指缩放 · 点选查看</span><span>{{ basemapNote }}</span>
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
        <RouterLink to="/atlas">查看图鉴 <ArrowRight :size="16" /></RouterLink>
      </div>
      <p v-if="focusNote" class="focus-hint" role="status">
        <LocateFixed :size="14" />{{ focusNote }}
      </p>
      <p v-if="atlas.state.loading" class="list-status">正在加载目录，完成后刷新列表与点位…</p>
      <div v-if="matches.length" class="discovery-cards">
        <SlideCard v-for="slide in visible" :key="slide.id" :slide="slide" compact />
      </div>
      <div v-else-if="!atlas.state.loading" class="empty-inline">
        没有匹配的点位，试试其他编号、名称或类型。
      </div>
      <div v-if="remaining" class="load-more-row">
        <button class="secondary-button" @click="loadMore">
          加载更多（还有 {{ remaining }} 条）
        </button>
        <span class="load-count">已显示 {{ visible.length }} / {{ matches.length }}</span>
      </div>
    </section>
    <p v-if="atlas.state.catalog.mode === 'demo'" class="page-footnote">
      示例目录用于体验浏览流程，不代表真实滑坡位置或分类。
    </p>
    <p v-else class="page-footnote">
      目录 {{ atlas.state.catalog.version }} · 记录来源与编号可在详情中核对。
    </p>
  </section>
</template>
