<script setup lang="ts">
import { computed, ref } from 'vue'
import { atlas, byType } from '../stores/atlas'
import { slideTypes, typeInfo, type SlideType } from '../types'

const type = ref<SlideType | ''>('')
const records = computed(() => atlas.state.catalog.records)
const typeCounts = computed(() => byType(records.value))
const current = computed(() =>
  type.value ? records.value.filter((slide) => slide.type === type.value) : records.value,
)
const latitudes = ['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N']
const latitudeBands = latitudes.slice(0, -1).map((start, index) => [start, latitudes[index + 1]])
const latitudeCounts = computed(() => {
  const counts = latitudeBands.map(() => 0)
  for (const slide of current.value) {
    const band = Math.min(counts.length - 1, Math.floor((slide.latitude + 90) / 30))
    counts[band] += 1
  }
  return counts
})
const tickIndexes = [4, 3, 2, 1, 0]
const latitudeStep = computed(() => {
  const target = Math.max(1, Math.max(...latitudeCounts.value) / tickIndexes[0])
  const unit = Math.max(1, 10 ** Math.floor(Math.log10(target)) / 2)
  return Math.ceil(target / unit) * unit
})
const latitudeMax = computed(() => tickIndexes[0] * latitudeStep.value)
const scopedAreas = computed(() =>
  current.value.flatMap((slide) => (slide.areaKm2 === null ? [] : [slide.areaKm2])),
)
const totalArea = computed(() => scopedAreas.value.reduce((sum, area) => sum + area, 0))
const countFormat = new Intl.NumberFormat('zh-CN')
const areaFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 })
const scope = computed(() => {
  const { mode, version } = atlas.state.catalog
  if (mode === 'demo') return '演示目录'
  if (version.startsWith('public-reference-')) return '公开样例'
  return '滑坡编目'
})
const share = (count: number, total: number) => (total ? (count / total) * 100 : 0)
const typeShare = (code: SlideType) => share(typeCounts.value[code] ?? 0, records.value.length)
const donutSegments = computed(() => {
  let cursor = 0
  return slideTypes.flatMap((code) => {
    const percent = typeShare(code)
    if (!percent) return []
    const start = cursor
    cursor += percent
    const color = typeInfo[code].color
    const fill =
      type.value && type.value !== code ? `color-mix(in srgb, ${color} 35%, var(--surface))` : color
    return [{ code, start, end: cursor, fill }]
  })
})
const donutFill = computed(() => {
  const segments = donutSegments.value
  if (segments.length < 2) return segments[0]?.fill ?? 'var(--line)'
  return `conic-gradient(${segments.map(({ fill, start, end }) => `${fill} ${start}% ${end}%`).join(', ')})`
})

function toggleType(code: SlideType) {
  type.value = type.value === code ? '' : code
}

function selectSector(event: MouseEvent) {
  const ring = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const radius = ring.width / 2
  const x = event.clientX - ring.left - radius
  const y = event.clientY - ring.top - radius
  const distance = Math.hypot(x, y) / radius
  if (distance < 0.67 || distance > 1) return
  // The CSS gradient starts at twelve o'clock and advances clockwise.
  const percent = ((Math.atan2(x, -y) / (2 * Math.PI) + 1) % 1) * 100
  const segment = donutSegments.value.find(({ start, end }) => percent >= start && percent < end)
  if (segment) toggleType(segment.code)
}
</script>

<template>
  <section class="content-view catalog-stats">
    <header class="stats-heading">
      <div>
        <p>{{ scope }}</p>
        <h1>滑坡目录统计</h1>
      </div>
      <RouterLink class="secondary-button" to="/">返回火星球</RouterLink>
    </header>

    <div class="stats-summary" aria-live="polite">
      <div>
        <span>当前记录</span
        ><strong data-testid="catalog-stats-total">{{ countFormat.format(current.length) }}</strong>
      </div>
      <div>
        <span>总面积 · km²</span
        ><strong>{{ scopedAreas.length ? areaFormat.format(totalArea) : '—' }}</strong>
      </div>
      <div>
        <span>平均面积 · km²</span
        ><strong>{{
          scopedAreas.length ? areaFormat.format(totalArea / scopedAreas.length) : '—'
        }}</strong>
      </div>
    </div>
    <p v-if="scopedAreas.length < current.length" class="stats-hint">
      面积统计使用有面积记录的滑坡。
    </p>

    <div class="stats-grid">
      <section class="stats-panel" aria-labelledby="type-chart-heading">
        <div class="stats-panel-heading">
          <h2 id="type-chart-heading">类型占比</h2>
          <button v-if="type" class="text-button" type="button" @click="type = ''">清除筛选</button>
        </div>
        <div class="type-chart">
          <div
            class="donut"
            aria-hidden="true"
            :class="{ interactive: records.length }"
            @click="selectSector"
          >
            <div class="donut-ring" :style="{ background: donutFill }"></div>
            <div class="donut-label">
              <strong>{{
                type ? typeShare(type).toFixed(1) + '%' : countFormat.format(records.length)
              }}</strong>
              <span>{{ type ? typeInfo[type].name : '全部记录' }}</span>
            </div>
          </div>
          <div class="type-legend" role="group" aria-label="按滑坡类型筛选">
            <button
              v-for="code in slideTypes"
              :key="code"
              class="stats-type"
              type="button"
              :style="{ '--type-color': typeInfo[code].color }"
              :aria-pressed="type === code"
              @click="toggleType(code)"
            >
              <span class="legend-dot" aria-hidden="true"></span>
              <span class="legend-type"
                >{{ typeInfo[code].name }} <small>{{ code }}</small></span
              >
              <span class="legend-value">
                <strong>{{ countFormat.format(typeCounts[code] ?? 0) }}</strong>
                <small>{{ typeShare(code).toFixed(1) }}%</small>
              </span>
            </button>
          </div>
        </div>
      </section>

      <section
        class="stats-panel"
        aria-labelledby="latitude-chart-heading"
        :style="{ '--latitude-color': type ? typeInfo[type].color : 'var(--accent)' }"
      >
        <div class="stats-panel-heading">
          <h2 id="latitude-chart-heading">纬度分布</h2>
          <span class="latitude-scope">{{ type ? typeInfo[type].name : '全部类型' }}</span>
        </div>
        <div class="latitude-chart">
          <div class="latitude-axis" aria-hidden="true">
            <span v-for="tick in tickIndexes" :key="tick">
              {{ countFormat.format(tick * latitudeStep) }}
            </span>
          </div>
          <div class="latitude-plot">
            <div class="latitude-grid" aria-hidden="true">
              <span v-for="tick in tickIndexes" :key="tick"></span>
            </div>
            <div
              v-for="(count, index) in latitudeCounts"
              :key="latitudeBands[index][0]"
              class="latitude-column"
              role="group"
              :aria-label="latitudeBands[index].join(' 至 ')"
              :style="{ '--bar-height': share(count, latitudeMax) + '%' }"
            >
              <strong data-testid="latitude-count">{{ countFormat.format(count) }}</strong>
              <span v-if="count" class="latitude-bar" aria-hidden="true"></span>
            </div>
          </div>
          <div class="latitude-labels" aria-hidden="true">
            <span v-for="[start, end] in latitudeBands" :key="start">
              <span>{{ start }}</span
              ><span>–{{ end }}</span>
            </span>
          </div>
          <div class="latitude-hemispheres" aria-hidden="true">
            <span>南纬</span><span>北纬</span>
          </div>
        </div>
      </section>
    </div>
  </section>
</template>

<style scoped>
.catalog-stats {
  width: min(100%, 1060px);
  margin-inline: auto;
  padding-bottom: 36px;
}
.stats-heading,
.stats-panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.stats-heading {
  align-items: flex-end;
  margin-bottom: 24px;
}
.stats-heading p,
.stats-hint,
.stats-panel-heading > span {
  color: var(--muted);
  font-size: 0.85rem;
}
.stats-heading h1 {
  margin: 4px 0 0;
  font-size: clamp(1.5rem, 5vw, 2.35rem);
}
.stats-heading a {
  white-space: nowrap;
}
.stats-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
}
.stats-summary div {
  display: grid;
  gap: 6px;
  padding: 16px clamp(10px, 2.6vw, 24px);
}
.stats-summary div + div {
  border-inline-start: 1px solid var(--line);
}
.stats-summary span {
  color: var(--muted);
  font-size: 0.77rem;
}
.stats-summary strong {
  color: var(--accent);
  font-size: clamp(0.92rem, 3.8vw, 1.55rem);
  font-variant-numeric: tabular-nums;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 350px), 1fr));
  gap: 16px;
  margin-top: 24px;
}
.stats-panel {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 17px clamp(12px, 2vw, 20px);
  min-width: 0;
  background: var(--surface);
}
.stats-panel-heading {
  min-height: 32px;
}
.stats-panel h2 {
  font-size: 1.13rem;
  margin: 0;
}
.type-chart {
  display: grid;
  grid-template-columns: minmax(98px, 0.9fr) minmax(140px, 1fr);
  align-items: center;
  gap: clamp(8px, 1.4vw, 18px);
  min-height: 256px;
  margin-top: 14px;
}
.donut {
  position: relative;
  width: 100%;
  max-width: 190px;
  aspect-ratio: 1;
  margin-inline: auto;
  display: grid;
  place-items: center;
  isolation: isolate;
}
.donut.interactive {
  cursor: pointer;
}
.donut-ring {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  mask: radial-gradient(circle closest-side, transparent 66%, #000 67% 99%, transparent 100%);
}
.donut-label {
  pointer-events: none;
  display: grid;
  justify-items: center;
  gap: 2px;
  z-index: 1;
}
.donut-label strong {
  font-size: clamp(0.99rem, 2.8vw, 1.73rem);
  font-variant-numeric: tabular-nums;
}
.donut-label span,
.legend-value small {
  color: var(--muted);
  font-size: 0.71rem;
}
.type-legend {
  display: grid;
  gap: 4px;
}
.stats-type {
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
  background: transparent;
  color: inherit;
  font: inherit;
  border: 1px solid transparent;
  border-radius: 6px;
  padding: 6px 8px;
  cursor: pointer;
  text-align: start;
}
.stats-type[aria-pressed='true'],
.stats-type:hover {
  border-color: color-mix(in srgb, var(--type-color) 65%, var(--surface));
  background: color-mix(in srgb, var(--type-color) 10%, var(--surface));
}
.stats-type:focus-visible {
  outline: 2px solid var(--type-color);
  outline-offset: 1px;
}
.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--type-color);
}
.legend-type,
.legend-value {
  display: grid;
  gap: 1px;
  font-size: 0.82rem;
  white-space: nowrap;
}
.legend-type small {
  color: var(--muted);
  font-size: 0.67rem;
  letter-spacing: 0.03em;
}
.legend-value {
  text-align: end;
  font-variant-numeric: tabular-nums;
}
.latitude-scope {
  white-space: nowrap;
}
.latitude-scope::before {
  content: '';
  display: inline-block;
  width: 6px;
  height: 6px;
  background: var(--latitude-color);
  border-radius: 50%;
  margin-inline-end: 6px;
  vertical-align: 0.15em;
}
.latitude-chart {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  column-gap: 9px;
  row-gap: 8px;
  margin-top: 34px;
}
.latitude-axis,
.latitude-grid {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.latitude-axis {
  margin-block: -0.55em;
  color: var(--muted);
  font-size: 0.71rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.1;
  text-align: end;
}
.latitude-plot,
.latitude-labels {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
}
.latitude-plot {
  position: relative;
  height: 176px;
}
.latitude-grid {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.latitude-grid span {
  border-bottom: 1px dashed var(--line);
}
.latitude-grid span:last-child {
  border-bottom-style: solid;
}
.latitude-column {
  position: relative;
  display: flex;
  align-items: end;
  justify-content: center;
}
.latitude-column:nth-child(5) {
  border-inline-start: 1px solid var(--line);
}
.latitude-column strong {
  position: absolute;
  bottom: calc(var(--bar-height) + 4px);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  color: var(--text);
  white-space: nowrap;
  transition: bottom 180ms ease;
}
.latitude-bar {
  width: min(65%, 34px);
  height: var(--bar-height);
  border-radius: 4px 4px 0 0;
  background: linear-gradient(
    var(--latitude-color),
    color-mix(in srgb, var(--latitude-color) 47%, var(--surface))
  );
  transition: height 180ms ease;
}
.latitude-labels,
.latitude-hemispheres {
  grid-column: 2;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.66rem;
  line-height: 1.25;
  white-space: nowrap;
}
.latitude-labels > span > span {
  display: block;
}
.latitude-hemispheres {
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin-top: 2px;
  color: var(--muted);
}
.latitude-hemispheres span {
  border-top: 1px solid var(--line);
  padding-top: 6px;
  margin-inline: 3px;
}
@media (max-width: 600px) {
  .type-chart {
    min-height: 210px;
    margin-top: 8px;
  }
}
@media (max-width: 375px) {
  .type-chart {
    grid-template-columns: minmax(0, 0.82fr) minmax(136px, 1fr);
  }
  .stats-type {
    gap: 5px;
    padding-inline: 5px;
  }
  .latitude-chart {
    column-gap: 6px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .latitude-column strong,
  .latitude-bar {
    transition: none;
  }
}
</style>
