<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Crosshair,
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
import { atlas } from '../stores/atlas'
import { filterSlides, findSlide, nearestSlide, neighborsOf } from '../lib/catalog'
import { parseSlideGeometry, type Ring } from '../lib/geometry'
import { apiBase, fetchSlideGeometry } from '../services/api'
import { formatMarsCoordinate, isMarsCoordinate } from '../lib/marsCoordinates'
import { coordinateLabel, slideTypes, typeInfo } from '../types'
import type { Slide, SlideType } from '../types'
const pageSize = 5
const formatCount = new Intl.NumberFormat('zh-CN').format
/** Records closer than this (degrees) can overlap on the globe. */
const overlapDegrees = 1.5
/** A coordinate within this distance is treated as "at" the nearest record. */
const coordinateMatchKm = 500
const query = ref('')
const type = ref<SlideType | ''>('')
const route = useRoute()
const globe = ref<InstanceType<typeof MarsGlobe>>()
const mapStage = ref<HTMLElement>()
const focusedId = ref('')
const focusNote = ref('')
const requestedPage = ref(0)
const basemap = ref<'loading' | 'ready' | 'fallback'>('loading')
const showGeometry = ref(true)
const coordinateOpen = ref(false)
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
const pageCount = computed(() => Math.max(1, Math.ceil(matches.value.length / pageSize)))
const page = computed(() => Math.min(requestedPage.value, pageCount.value - 1))
const pageStart = computed(() => page.value * pageSize)
const visible = computed(() => matches.value.slice(pageStart.value, pageStart.value + pageSize))
const focusedRecord = computed(
  () =>
    matches.value.find((item) => item.id === focusedId.value) ??
    atlas.state.catalog.records.find((item) => item.id === focusedId.value),
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
    blocked:
      atlas.state.catalog.mode === 'reference'
        ? '边界几何：当前目录暂不提供叠加'
        : '边界几何：坐标系未确认，暂不叠加',
    missing: '边界几何：当前记录暂无',
    offline: '边界几何：未连接数据服务',
  }[geometryState.value]
})
watch([query, type], () => {
  requestedPage.value = 0
  focusNote.value = ''
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
function revealRecord(id: string): void {
  const index = matches.value.findIndex((record) => record.id === id)
  if (index >= 0) requestedPage.value = Math.floor(index / pageSize)
}
function applyFocus(record: Slide, note?: string): void {
  focusedId.value = record.id
  focusNote.value = note ?? `已定位 ${record.name}`
  neighbors.value = neighborsOf(matches.value, record.id, overlapDegrees)
  neighborIndex.value = -1
  globe.value?.focusPoint(record.id)
  revealRecord(record.id)
}
/**
 * Filters reset the directory to the first page, and that watcher runs after
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
  const record = matches.value.find((item) => item.id === id)
  if (record) applyFocus(record)
}
function locateFromList(id: string) {
  select(id)
  mapStage.value?.scrollIntoView({ block: 'nearest' })
}
function cycleNeighbor() {
  const next = neighbors.value[neighborIndex.value + 1]
  if (!next) return
  neighborIndex.value += 1
  focusedId.value = next.id
  focusNote.value = `已定位 ${next.name}（相邻点位）`
  globe.value?.focusPoint(next.id)
  revealRecord(next.id)
}
function reset() {
  focusedId.value = ''
  focusNote.value = ''
  neighbors.value = []
  neighborIndex.value = -1
  globe.value?.resetView()
}
function turnPage(direction: number) {
  requestedPage.value = Math.max(0, Math.min(page.value + direction, pageCount.value - 1))
}
function clearFilters() {
  query.value = ''
  type.value = ''
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
      <div class="explore-actions">
        <RouterLink class="secondary-button" to="/statistics">目录统计</RouterLink>
      </div>
    </div>
    <div ref="mapStage" class="globe-stage">
      <div class="globe-topline">
        <span>{{ type ? typeInfo[type].name : '全球视野' }}</span>
        <span v-if="atlas.state.catalog.mode === 'demo'">演示数据</span>
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
    <section class="catalog-panel" aria-labelledby="explore-catalog-title">
      <div class="catalog-toolbar">
        <div class="catalog-heading">
          <h2 id="explore-catalog-title">滑坡目录</h2>
          <RouterLink class="catalog-atlas-link" to="/atlas">
            典型案例 <ArrowUpRight :size="15" aria-hidden="true" />
          </RouterLink>
        </div>
        <div class="search-box catalog-search">
          <Search :size="17" aria-hidden="true" />
          <input
            v-model="query"
            aria-label="搜索滑坡编号、名称或说明"
            placeholder="搜索滑坡编号或名称"
            inputmode="search"
            @keydown.esc="query = ''"
          />
          <button
            v-if="query"
            type="button"
            class="icon-button"
            aria-label="清除搜索"
            @click="query = ''"
          >
            <X :size="15" />
          </button>
        </div>
        <div class="catalog-filters" role="group" aria-label="类型筛选">
          <button type="button" :aria-pressed="!type" @click="type = ''">全部</button>
          <button
            v-for="t in slideTypes"
            :key="t"
            type="button"
            :aria-pressed="type === t"
            :style="{ '--filter-color': typeInfo[t].color }"
            @click="type = type === t ? '' : t"
          >
            {{ typeInfo[t].name }}
          </button>
        </div>
      </div>
      <div v-if="focusNote" class="catalog-focus">
        <p role="status"><LocateFixed :size="14" aria-hidden="true" /> {{ focusNote }}</p>
        <button
          v-if="neighbors.length > neighborIndex + 1"
          type="button"
          class="catalog-neighbor"
          @click="cycleNeighbor"
        >
          下一相邻点位（{{ neighborIndex + 2 }}/{{ neighbors.length }}）
          <ChevronRight :size="13" />
        </button>
        <span v-else-if="neighbors.length">已到最后一个相邻点位</span>
      </div>
      <p v-if="atlas.state.loading" class="catalog-loading" role="status">正在更新目录…</p>
      <ul v-if="visible.length" id="explore-catalog-list" class="catalog-list">
        <li
          v-for="slide in visible"
          :key="slide.id"
          class="catalog-row"
          :class="{ 'is-located': slide.id === focusedId }"
          :style="{ '--entry-color': typeInfo[slide.type].color }"
          :data-slide-id="slide.id"
        >
          <RouterLink
            class="catalog-record-link"
            :to="{ name: 'catalog-record', params: { id: slide.id } }"
            :aria-label="`查看${slide.name}详情`"
          >
            <span class="catalog-place">
              <span class="catalog-record-title">
                <strong>{{ slide.name }}</strong>
                <ChevronRight :size="13" aria-hidden="true" />
              </span>
              <span class="catalog-coordinates">
                {{ coordinateLabel(slide.longitude, slide.latitude) }}
              </span>
            </span>
            <span class="catalog-kind">
              <span>{{ typeInfo[slide.type].name }}</span>
              <span v-if="slide.areaKm2 !== null">{{ slide.areaKm2.toFixed(2) }} km²</span>
            </span>
          </RouterLink>
          <button
            class="icon-button catalog-locate"
            type="button"
            :aria-label="`定位${slide.name}`"
            :aria-pressed="slide.id === focusedId"
            title="在火星球上定位"
            @click="locateFromList(slide.id)"
          >
            <Crosshair :size="18" aria-hidden="true" />
          </button>
        </li>
      </ul>
      <div v-else-if="!atlas.state.loading" class="catalog-empty">
        <p role="status">没有匹配的滑坡</p>
        <button type="button" @click="clearFilters">清除筛选</button>
      </div>
      <div v-if="visible.length" class="catalog-pagination">
        <span role="status">
          {{ formatCount(pageStart + 1) }}–{{ formatCount(pageStart + visible.length) }} /
          {{ formatCount(matches.length) }} 条
        </span>
        <button
          v-if="pageCount > 1"
          class="icon-button"
          type="button"
          aria-label="上一页"
          aria-controls="explore-catalog-list"
          :disabled="page === 0"
          @click="turnPage(-1)"
        >
          <ChevronLeft :size="18" />
        </button>
        <button
          v-if="pageCount > 1"
          class="icon-button"
          type="button"
          aria-label="下一页"
          aria-controls="explore-catalog-list"
          :disabled="page + 1 === pageCount"
          @click="turnPage(1)"
        >
          <ChevronRight :size="18" />
        </button>
      </div>
    </section>
  </section>
</template>

<style scoped>
.explore-actions {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.explore-actions a {
  white-space: nowrap;
}
.catalog-panel {
  min-width: 0;
  align-self: start;
  margin-block: var(--space-lg) 0;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--surface);
}
.catalog-toolbar {
  display: grid;
  gap: var(--space-xs);
  padding: var(--space-xs) var(--space-md) var(--space-sm);
}
.catalog-heading,
.catalog-atlas-link,
.catalog-pagination {
  display: flex;
  align-items: center;
}
.catalog-heading {
  justify-content: space-between;
  gap: var(--space-xs);
}
.catalog-heading h2 {
  font-size: 1.125rem;
  font-weight: 600;
}
.catalog-atlas-link {
  gap: var(--space-2xs);
  min-height: 44px;
  color: var(--accent-hover);
  font-size: var(--text-sm);
  white-space: nowrap;
}
.catalog-atlas-link:hover {
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
.catalog-search {
  height: 44px;
  padding-inline: var(--space-sm) 0;
  border-color: var(--line);
  border-radius: 9px;
  background: var(--bg);
}
.catalog-search input {
  font-size: 1rem;
  padding-inline-end: var(--space-xs);
}
.catalog-search .icon-button {
  flex: 0 0 44px;
}
.catalog-filters {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: var(--space-2xs);
}
.catalog-filters button {
  --filter-color: var(--accent-hover);
  min-height: 44px;
  padding: 0;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--muted);
  font-size: 0.8rem;
}
.catalog-filters button:hover {
  background: var(--surface-raised);
  color: var(--text);
}
.catalog-filters button[aria-pressed='true'] {
  border-color: color-mix(in oklab, var(--filter-color) 35%, var(--surface));
  background: color-mix(in oklab, var(--filter-color) 12%, var(--surface));
  color: var(--filter-color);
  font-weight: 600;
}
.catalog-list {
  list-style: none;
  margin: 0;
  padding: 0 var(--space-xs);
  border-block: 1px solid var(--line);
  background: color-mix(in oklab, var(--bg) 45%, var(--surface));
}
.catalog-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 44px;
  align-items: center;
  gap: var(--space-2xs);
  padding-inline-end: var(--space-2xs);
}
.catalog-row + .catalog-row {
  border-top: 1px solid var(--line);
}
.catalog-row.is-located {
  background: var(--selected-bg);
  box-shadow: inset 2px 0 var(--accent);
}
.catalog-record-link {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--space-xs);
  min-height: 57px;
  padding: var(--space-2xs) var(--space-xs);
  border-radius: var(--radius-control);
}
.catalog-record-link:hover {
  background: var(--surface-raised);
}
.catalog-place,
.catalog-kind {
  display: grid;
  gap: 2px;
  min-width: 0;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.catalog-record-title {
  display: flex;
  align-items: center;
  gap: var(--space-2xs);
}
.catalog-record-title strong {
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: var(--text-sm);
  font-weight: 600;
}
.catalog-record-title svg,
.catalog-coordinates,
.catalog-kind > span + span {
  color: var(--muted);
  font-size: var(--text-xs);
}
.catalog-kind {
  justify-items: end;
  margin-inline-end: var(--space-2xs);
  color: var(--entry-color);
  font-size: 0.8rem;
}
.catalog-locate {
  color: var(--text-secondary);
  border-radius: 8px;
}
.catalog-locate:hover,
.catalog-locate[aria-pressed='true'] {
  color: var(--accent-hover);
  background: var(--surface-raised);
}
.catalog-focus,
.catalog-loading,
.catalog-empty {
  padding-inline: var(--space-md);
  font-size: var(--text-sm);
}
.catalog-focus {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0 var(--space-xs);
  padding-block: var(--space-2xs);
  border-top: 1px solid var(--line);
  color: var(--accent-hover);
}
.catalog-focus p,
.catalog-neighbor {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2xs);
}
.catalog-neighbor {
  min-height: 44px;
  margin-inline-start: auto;
  padding: 0;
}
.catalog-neighbor:hover,
.catalog-empty button:hover {
  text-decoration: underline;
}
.catalog-focus > span,
.catalog-loading,
.catalog-empty p {
  color: var(--muted);
}
.catalog-loading {
  padding-block: var(--space-xs);
}
.catalog-empty {
  padding-block: var(--space-md);
  border-top: 1px solid var(--line);
  text-align: center;
}
.catalog-empty button {
  min-height: 44px;
  color: var(--accent-hover);
}
.catalog-pagination {
  gap: var(--space-2xs);
  min-height: 56px;
  padding-inline: var(--space-md) var(--space-xs);
}
.catalog-pagination > span {
  margin-inline-end: auto;
  color: var(--muted);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}
.catalog-pagination .icon-button {
  flex: 0 0 44px;
  border-radius: 8px;
  color: var(--text-secondary);
}
@media (min-width: 900px) {
  .catalog-panel {
    margin-block-start: 0;
  }
}
</style>
