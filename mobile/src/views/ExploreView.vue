<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import {
  ArrowRight,
  ChevronRight,
  LocateFixed,
  MapPin,
  Minus,
  Plus,
  Search,
  Shapes,
  X,
} from '@lucide/vue'
import { useRoute } from 'vue-router'
import MarsGlobe from '../components/MarsGlobe.vue'
import SlideCard from '../components/SlideCard.vue'
import CardRail from '../components/CardRail.vue'
import { atlas } from '../stores/atlas'
import { crowdedSlideIds, filterSlides, findSlide, nearestSlide, neighborsOf } from '../lib/catalog'
import { parseSlideGeometry, type Ring } from '../lib/geometry'
import { apiBase, fetchSlideGeometry } from '../services/api'
import { formatMarsCoordinate, isMarsCoordinate } from '../lib/marsCoordinates'
import { slideTypes, typeInfo } from '../types'
import type { Slide, SlideType } from '../types'
const pageSize = 20
/** Records closer than this (degrees) can overlap on the globe. */
const overlapDegrees = 1.5
/** A coordinate within this distance is treated as "at" the nearest record. */
const coordinateMatchKm = 500
const query = ref('')
const type = ref<SlideType | ''>('')
const route = useRoute()
const globe = ref<InstanceType<typeof MarsGlobe>>()
const focusedId = ref('')
const focusNote = ref('')
const visibleCount = ref(pageSize)
const basemap = ref<'loading' | 'ready' | 'fallback'>('loading')
const showGeometry = ref(true)
const coordinateOpen = ref(false)
const searchOpen = ref(false)
const longitudeInput = ref('')
const latitudeInput = ref('')
const coordinateNote = ref('')
const neighbors = ref<Slide[]>([])
/** Index of the neighbor currently shown; -1 means the focused record itself. */
const neighborIndex = ref(-1)
const geometryRing = ref<Ring | null>(null)
const geometryState = ref<'idle' | 'loading' | 'ready' | 'blocked' | 'missing' | 'offline'>('idle')
let geometryRequest = 0
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
const visible = computed(() => matches.value.slice(0, visibleCount.value))
const remaining = computed(() => Math.max(0, matches.value.length - visible.value.length))
const focusedRecord = computed(
  () =>
    matches.value.find((item) => item.id === focusedId.value) ??
    atlas.state.catalog.records.find((item) => item.id === focusedId.value),
)
const crowdedCount = computed(() => crowdedSlideIds(matches.value, overlapDegrees).size)
const modeLabel = computed(() =>
  atlas.state.catalog.mode === 'reference'
    ? '原始坐标 · 待核准'
    : atlas.state.catalog.mode === 'catalog'
      ? '真实目录'
      : '演示数据',
)
const basemapNote = computed(
  () =>
    ({
      loading: '正在加载全球底图…',
      ready: '底图：Solar System Scope（CC BY 4.0）',
      fallback: '底图未加载 · 点位按经纬度显示',
    })[basemap.value],
)
const geometryLabel = computed(() => {
  if (!focusedRecord.value) return '边界几何：未选点位'
  return {
    idle: '边界几何：未选点位',
    loading: '边界几何：正在加载…',
    ready: showGeometry.value ? '边界几何：已显示' : '边界几何：已隐藏',
    blocked: '边界几何：坐标系未确认，暂不叠加',
    missing: '边界几何：当前记录暂无',
    offline: '边界几何：未连接数据服务',
  }[geometryState.value]
})
watch([query, type], () => {
  visibleCount.value = pageSize
})
watch(
  focusedId,
  async (id) => {
    geometryRequest += 1
    const request = geometryRequest
    geometryRing.value = null
    if (!id) {
      geometryState.value = 'idle'
      return
    }
    if (atlas.state.catalog.mode === 'reference') {
      geometryState.value = 'blocked'
      return
    }
    if (!apiBase) {
      geometryState.value = 'offline'
      return
    }
    geometryState.value = 'loading'
    try {
      const payload = await fetchSlideGeometry(id)
      if (request !== geometryRequest) return
      if (!payload) {
        geometryState.value = 'missing'
        return
      }
      const parsed = parseSlideGeometry(payload)
      if (!parsed) {
        geometryState.value = 'missing'
        return
      }
      if (!parsed.canOverlay) {
        geometryState.value = 'blocked'
        return
      }
      geometryRing.value = parsed.ring
      geometryState.value = 'ready'
    } catch {
      if (request === geometryRequest) geometryState.value = 'offline'
    }
  },
  { immediate: true },
)
function revealCard(id: string): void {
  const index = matches.value.findIndex((item) => item.id === id)
  if (index >= visibleCount.value) visibleCount.value = index + 1
  void nextTick(() => {
    document
      .querySelector(`[data-slide-id="${CSS.escape(id)}"]`)
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  })
}
function applyFocus(record: Slide, note?: string): void {
  focusedId.value = record.id
  focusNote.value = note ?? `已定位 ${record.sourceId ?? record.id} · ${record.name}`
  neighbors.value = neighborsOf(matches.value, record.id, overlapDegrees)
  neighborIndex.value = -1
  globe.value?.focusPoint(record.id)
  revealCard(record.id)
}
/**
 * Filters reset `visibleCount` to the first page, and that watcher runs after
 * the caller. Wait for it before revealing a record that may sit deeper.
 */
async function resetFiltersForFocus(): Promise<void> {
  if (!query.value && !type.value) return
  query.value = ''
  type.value = ''
  await nextTick()
}
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
  await resetFiltersForFocus()
  applyFocus(record)
}
watch(() => route.fullPath, focusFromRoute)
watch(() => [atlas.state.loading, atlas.state.catalog.version], focusFromRoute)
onMounted(async () => {
  await focusFromRoute()
  if (!route.query.focus && atlas.state.catalog.records[0])
    globe.value?.focusPoint(atlas.state.catalog.records[0].id)
})
function select(id: string) {
  focusedId.value = id
  const record = matches.value.find((item) => item.id === id)
  if (record) {
    focusNote.value = `已定位 ${record.sourceId ?? record.id} · ${record.name}`
    neighbors.value = neighborsOf(matches.value, id, overlapDegrees)
    neighborIndex.value = -1
    revealCard(id)
  }
  atlas.select(id)
}
function cycleNeighbor() {
  const next = neighbors.value[neighborIndex.value + 1]
  if (!next) return
  neighborIndex.value += 1
  focusedId.value = next.id
  focusNote.value = `已定位 ${next.sourceId ?? next.id} · ${next.name}（相邻点位）`
  globe.value?.focusPoint(next.id)
  revealCard(next.id)
}
function reset() {
  focusedId.value = ''
  neighbors.value = []
  neighborIndex.value = -1
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
function clearCoordinateNote() {
  coordinateNote.value = ''
}
async function locateCoordinate() {
  const longitude = Number(longitudeInput.value)
  const latitude = Number(latitudeInput.value)
  if (
    !longitudeInput.value.trim() ||
    !latitudeInput.value.trim() ||
    !isMarsCoordinate({ longitude, latitude })
  ) {
    coordinateNote.value = '请输入有效坐标：经度 -180～180，纬度 -90～90。'
    return
  }
  coordinateNote.value = ''
  const near = nearestSlide(atlas.state.catalog.records, longitude, latitude)
  globe.value?.focusCoordinate(longitude, latitude)
  if (!near || near.distanceKm > coordinateMatchKm) {
    focusedId.value = ''
    neighbors.value = []
    neighborIndex.value = -1
    focusNote.value = `已定位坐标 ${formatMarsCoordinate({ longitude, latitude })}`
    coordinateNote.value = near
      ? `坐标 ${coordinateMatchKm} km 内没有目录记录（最近约 ${Math.round(near.distanceKm)} km）。`
      : '当前目录没有可比较的记录。'
    return
  }
  await resetFiltersForFocus()
  applyFocus(
    near.slide,
    `坐标 ${formatMarsCoordinate({ longitude, latitude })} 最近记录 ${
      near.slide.sourceId ?? near.slide.id
    } · 约 ${Math.round(near.distanceKm)} km`,
  )
}
</script>
<template>
  <section class="explore-view">
    <div class="explore-heading">
      <div>
        <h1>探索火星</h1>
      </div>
      <button
        class="icon-button search-toggle"
        aria-label="搜索与筛选点位"
        :aria-expanded="searchOpen"
        @click="searchOpen = !searchOpen"
      >
        <Search :size="20" />
      </button>
    </div>
    <div v-show="searchOpen || query || type" class="explore-search">
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
        <button :class="{ active: !type }" :aria-pressed="!type" @click="clearFilters()">
          全部</button
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
    </div>
    <div class="globe-stage">
      <div class="globe-topline">
        <span><i></i>火星 · {{ type || '全球视野' }}</span
        ><span>{{ modeLabel }}</span>
      </div>
      <MarsGlobe
        ref="globe"
        :points="matches"
        :selected-id="focusedId"
        :geometry="geometryRing"
        :show-geometry="showGeometry"
        @select="select"
        @basemap="basemap = $event"
      />
      <div class="globe-tools">
        <button class="icon-button" aria-label="放大火星球" @click="globe?.zoomIn()">
          <Plus :size="20" /></button
        ><button class="icon-button" aria-label="缩小火星球" @click="globe?.zoomOut()">
          <Minus :size="20" /></button
        ><button
          v-if="geometryState === 'ready'"
          class="icon-button"
          :disabled="geometryState !== 'ready'"
          aria-label="显示或隐藏当前记录的边界几何"
          :aria-pressed="showGeometry"
          :title="geometryLabel"
          @click="showGeometry = !showGeometry"
        >
          <Shapes :size="19" /></button
        ><button
          class="icon-button"
          aria-label="按坐标定位"
          :aria-pressed="coordinateOpen"
          @click="coordinateOpen = !coordinateOpen"
        >
          <MapPin :size="19" /></button
        ><button class="icon-button" aria-label="复位火星球" @click="reset">
          <LocateFixed :size="19" />
        </button>
      </div>
      <p
        v-if="focusedRecord && geometryState === 'ready'"
        class="globe-geometry-status"
        aria-live="polite"
      >
        {{ geometryLabel }}
      </p>
      <form v-if="coordinateOpen" class="coordinate-panel" @submit.prevent="locateCoordinate">
        <label>
          经度
          <input
            v-model="longitudeInput"
            type="text"
            inputmode="decimal"
            autocomplete="off"
            placeholder="-180～180"
            @input="clearCoordinateNote"
          />
        </label>
        <label>
          纬度
          <input
            v-model="latitudeInput"
            type="text"
            inputmode="decimal"
            autocomplete="off"
            placeholder="-90～90"
            @input="clearCoordinateNote"
          />
        </label>
        <button type="submit" class="secondary-button">定位</button>
        <p v-if="coordinateNote" class="coordinate-note" role="status">{{ coordinateNote }}</p>
      </form>
      <div class="globe-footer">
        <span>{{ basemapNote }}</span>
      </div>
    </div>
    <section class="discovery-section">
      <div class="section-heading">
        <div>
          <h2>
            {{ query || type ? '查询结果' : '滑坡目录' }} <span>{{ matches.length }}</span>
          </h2>
        </div>
        <RouterLink to="/atlas#samples">查看全部 <ArrowRight :size="16" /></RouterLink>
      </div>
      <p v-if="focusNote" class="focus-hint" role="status">
        <LocateFixed :size="14" /><span>{{ focusNote }}</span
        ><button
          v-if="neighbors.length > neighborIndex + 1"
          class="text-button"
          @click="cycleNeighbor"
        >
          下一个相邻点位（{{ neighborIndex + 2 }}/{{ neighbors.length }}）
          <ChevronRight :size="13" />
        </button>
        <span v-else-if="neighbors.length" class="focus-hint-note">
          {{
            neighborIndex < 0
              ? `附近共 ${neighbors.length} 个密集点位`
              : `已到最后一个相邻点位（共 ${neighbors.length} 个）`
          }}
        </span>
      </p>
      <p v-if="atlas.state.loading" class="list-status">正在加载目录，完成后刷新列表与点位…</p>
      <CardRail
        v-if="matches.length"
        :key="`${query}:${type}`"
        :count="visible.length"
        label="滑坡目录"
      >
        <SlideCard
          v-for="slide in visible"
          :key="slide.id"
          :slide="slide"
          compact
          :highlighted="slide.id === focusedId"
        />
      </CardRail>
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
    <details class="quiet-details">
      <summary>数据说明</summary>
      <p>
        当前展示 {{ atlas.state.catalog.records.length }} 条公开记录。{{
          atlas.state.catalog.mode === 'reference'
            ? '球面沿用原始坐标数值，纬度约定待核准；没有做纬度转换，不用于精确定位。'
            : ''
        }}来源见记录详情。
      </p>
      <p v-if="crowdedCount">相邻点位可能重叠，选中后可依次查看。</p>
    </details>
  </section>
</template>
