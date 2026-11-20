<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ArrowLeft, Bookmark, Crosshair, ExternalLink } from '@lucide/vue'
import { useRoute } from 'vue-router'
import ZoomableImage from '../components/ZoomableImage.vue'
import { atlasContent, caseById, caseTitle } from '../lib/atlas'
import { findSlide } from '../lib/catalog'
import { caseImageUrl, sampleById } from '../lib/samples'
import { apiBase } from '../services/api'
import { atlas } from '../stores/atlas'
import { typeInfo } from '../types'

const props = defineProps<{ id: string; caseKey?: string }>()
const route = useRoute()
const fromAtlas = computed(() => Boolean(props.caseKey) || route.query.from === 'atlas')
const record = computed(() => findSlide(atlas.state.catalog.records, props.id))
const inventoryPaper = atlasContent.sourceDocument
const isGeoMarsInventory = computed(() => record.value?.source.url === inventoryPaper.url)
const caseInfo = computed(() => {
  if (props.caseKey) return caseById(props.caseKey)
  if (fromAtlas.value) {
    const id = record.value?.id ?? props.id
    return atlasContent.cases.find((item) => item.landslideId === id)
  }
})
const caseHeading = computed(() => (caseInfo.value ? caseTitle(caseInfo.value) : ''))
const backLink = computed(() => {
  const kind = caseInfo.value?.type
  if (kind && route.query.fromType === kind)
    return { path: `/atlas/type/${kind}`, label: `返回${typeInfo[kind].name}` }
  if (fromAtlas.value) return { path: '/atlas', label: '返回图鉴' }
  if (route.query.from === 'chat') return { path: '/chat', label: '返回助手' }
  if (route.query.from === 'mine') return { path: '/mine', label: '返回我的' }
  return { path: '/', label: '返回火星球' }
})
const numberFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 4 })
function formatCoordinate(value: number, positive: string, negative: string): string {
  const degrees = `${Math.abs(value).toFixed(4)}°`
  if (value === 0) return degrees
  return `${value < 0 ? negative : positive} ${degrees}`
}
const failedImage = ref(false)
const sample = computed(() =>
  fromAtlas.value && record.value ? sampleById(record.value.id) : undefined,
)
const caseImage = computed(() => (caseInfo.value ? caseImageUrl(record.value, apiBase) : undefined))
watch(caseImage, () => {
  failedImage.value = false
})
</script>

<template>
  <section
    class="content-view catalog-record-view"
    :class="{ 'is-case': caseImage && !failedImage }"
  >
    <RouterLink class="text-button back-link" :to="backLink.path">
      <ArrowLeft :size="16" />{{ backLink.label }}
    </RouterLink>

    <header v-if="record || caseInfo" class="record-header">
      <div>
        <h1>
          {{ record?.sourceId != null ? `滑坡 #${record.sourceId}` : record?.name || caseHeading }}
        </h1>
        <p v-if="caseHeading && record">{{ caseHeading }}</p>
      </div>
      <span
        v-if="record"
        class="record-class"
        :style="{ '--record-color': typeInfo[record.type].color }"
      >
        {{ typeInfo[record.type].name }}（{{ record.type }}）
      </span>
    </header>

    <div v-if="record" class="record-overview">
      <section
        v-if="caseImage && !failedImage"
        class="case-photo"
        aria-labelledby="case-photo-title"
      >
        <div class="case-photo-heading">
          <h2 id="case-photo-title">案例影像</h2>
          <RouterLink v-if="sample" class="text-button" :to="`/recognize?sample=${sample.id}`"
            >识别这张影像</RouterLink
          >
        </div>
        <ZoomableImage
          square
          :src="caseImage"
          :alt="caseHeading || record.name"
          @error.capture="failedImage = true"
        />
      </section>
      <div class="record-data">
        <section class="record-section" aria-labelledby="record-data-title">
          <h2 id="record-data-title">编目信息</h2>
          <dl class="record-facts">
            <div>
              <dt>原始编号</dt>
              <dd>{{ record.sourceId ?? record.id }}</dd>
            </div>
            <div>
              <dt>类型</dt>
              <dd>{{ typeInfo[record.type].name }}（{{ record.type }}）</dd>
            </div>
            <div>
              <dt>面积</dt>
              <dd>
                {{
                  record.areaKm2 === null
                    ? '暂无记录'
                    : `${numberFormat.format(record.areaKm2)} km²`
                }}
              </dd>
            </div>
            <div>
              <dt>周长</dt>
              <dd>
                {{
                  record.perimeterKm == null
                    ? '暂无记录'
                    : `${numberFormat.format(record.perimeterKm)} km`
                }}
              </dd>
            </div>
            <div>
              <dt>经度</dt>
              <dd>{{ formatCoordinate(record.longitude, '东经', '西经') }}</dd>
            </div>
            <div>
              <dt>纬度</dt>
              <dd>{{ formatCoordinate(record.latitude, '北纬', '南纬') }}</dd>
            </div>
            <div v-if="caseInfo">
              <dt>CTX 图幅</dt>
              <dd>{{ caseInfo.tile }}</dd>
            </div>
            <div v-if="isGeoMarsInventory">
              <dt>参考论文</dt>
              <dd>
                <a :href="inventoryPaper.url" target="_blank" rel="noopener noreferrer">
                  《{{ inventoryPaper.title }}》<ExternalLink :size="13" />
                </a>
              </dd>
            </div>
          </dl>
        </section>

        <div class="record-actions">
          <RouterLink class="primary-button" :to="{ path: '/', query: { focus: record.id } }">
            <Crosshair :size="18" />定位到火星球
          </RouterLink>
          <button
            class="secondary-button"
            :aria-pressed="atlas.isSaved(record.id)"
            @click="atlas.toggleSaved(record.id)"
          >
            <Bookmark :size="17" :fill="atlas.isSaved(record.id) ? 'currentColor' : 'none'" />
            {{ atlas.isSaved(record.id) ? '已收藏' : '收藏' }}
          </button>
        </div>
      </div>
    </div>
    <p v-else class="notice" role="status">
      {{ atlas.state.loading ? '正在加载编目…' : '当前编目中没有这条记录。' }}
    </p>

    <section
      v-if="caseInfo"
      class="record-section record-description"
      aria-labelledby="case-notes-title"
    >
      <h2 id="case-notes-title">案例观察</h2>
      <p v-if="(!caseImage || failedImage) && !atlas.state.loading" class="muted">暂无公开影像</p>
      <p>{{ caseInfo.subtitle }}</p>
      <ul>
        <li v-for="point in caseInfo.observations" :key="point">{{ point }}</li>
      </ul>
      <p class="record-citation">{{ caseInfo.citationLocator }}</p>
    </section>
    <section v-else-if="record?.summary" class="record-section record-description">
      <h2>记录说明</h2>
      <p>{{ record.summary }}</p>
    </section>
  </section>
</template>

<style scoped>
.catalog-record-view {
  width: min(100%, 760px);
  margin-inline: auto;
  padding-bottom: 32px;
}
.catalog-record-view.is-case {
  width: min(100%, 1040px);
}
.record-header {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 10px 18px;
  margin-block: 15px 23px;
}
.record-header h1 {
  margin: 0;
  font-size: clamp(1.6rem, 5vw, 2.5rem);
}
.record-header p {
  margin: 6px 0 0;
  color: var(--text-secondary);
}
.record-class {
  border: 1px solid color-mix(in srgb, var(--record-color) 45%, var(--surface));
  border-radius: 5px;
  padding: 5px 9px;
  color: var(--record-color);
  font-size: 0.83rem;
}
.record-section {
  padding: 17px clamp(13px, 3vw, 24px);
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--surface);
}
.record-section h2 {
  margin: 0 0 9px;
  font-size: 1.13rem;
}
.record-overview {
  display: grid;
  align-items: start;
  gap: 20px;
}
.case-photo,
.record-data {
  min-width: 0;
}
.case-photo {
  width: min(100%, 510px);
  margin-inline: auto;
}
.case-photo-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 28px;
  margin-bottom: 8px;
}
.case-photo-heading h2 {
  font-size: 1.12rem;
  margin: 0;
}
.case-photo-heading .text-button {
  font-size: 0.82rem;
}
.record-facts {
  margin: 0;
}
.record-facts > div {
  display: grid;
  grid-template-columns: minmax(65px, 0.7fr) minmax(0, 1.3fr);
  align-items: baseline;
  gap: 12px;
  padding: 11px 0;
  border-top: 1px solid var(--line);
}
.record-facts dt {
  color: var(--muted);
}
.record-facts dd {
  margin: 0;
  text-align: end;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
.record-facts a {
  color: var(--accent);
}
.record-facts a svg {
  display: inline-block;
  vertical-align: -0.14em;
}
.record-citation {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 0.82rem;
}
.record-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}
.record-description {
  margin-top: 14px;
}
.record-description p,
.record-description li {
  line-height: 1.65;
}
.record-description ul {
  padding-inline-start: 1.2em;
}
@media (min-width: 820px) {
  .is-case .record-overview {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.03fr);
    column-gap: 25px;
  }
}
</style>
