<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeft, ArrowUpRight, BookOpen, ChevronDown, ChevronRight } from '@lucide/vue'
import { atlasContent, caseRecord, casesOfType, caseTitle, typeEntry } from '../lib/atlas'
import { caseImageUrl } from '../lib/samples'
import { apiBase } from '../services/api'
import { atlas } from '../stores/atlas'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'

const props = defineProps<{ type: string }>()
const type = computed(() => props.type.toUpperCase() as SlideType)
const entry = computed(() => typeEntry(type.value))
const cases = computed(() => casesOfType(type.value))
const casesWithImage = computed(
  () =>
    new Set(
      cases.value
        .filter((item) => caseImageUrl(caseRecord(item, atlas.state.catalog.records), apiBase))
        .map((item) => item.id),
    ),
)
const comparison = computed(() =>
  atlasContent.comparisons.find((pair) => pair.left === type.value || pair.right === type.value),
)
</script>

<template>
  <section
    class="content-view atlas-type-page"
    :style="entry ? { '--guide-color': typeInfo[entry.type].color } : undefined"
  >
    <div class="type-page-links">
      <RouterLink class="type-page-back" to="/atlas">
        <ArrowLeft :size="16" aria-hidden="true" />返回图鉴
      </RouterLink>
      <RouterLink
        v-if="entry"
        class="type-page-compare"
        :to="{ path: '/atlas/compare', query: comparison ? { pair: comparison.id } : {} }"
      >
        类型对比 <ArrowUpRight :size="15" aria-hidden="true" />
      </RouterLink>
    </div>

    <p v-if="!entry" class="notice">未知的滑坡类别：{{ props.type }}</p>
    <div v-else class="type-guide">
      <header class="type-hero">
        <div class="type-title">
          <h1>{{ entry.name }}</h1>
          <span class="type-abbreviation">{{ entry.type }}</span>
        </div>
        <p>{{ entry.quote }}</p>
      </header>

      <nav class="type-guide-nav" aria-label="滑坡类型">
        <RouterLink
          v-for="kind in slideTypes"
          :key="kind"
          :to="`/atlas/type/${kind}`"
          :class="{ current: kind === type }"
        >
          {{ typeInfo[kind].name }}
        </RouterLink>
      </nav>

      <div class="type-guide-grid">
        <section class="type-signals" aria-labelledby="type-signals-title">
          <h2 id="type-signals-title">识别要点</h2>
          <ul>
            <li v-for="point in entry.recognize" :key="point">{{ point }}</li>
          </ul>
        </section>

        <section class="type-cases" aria-labelledby="type-cases-title">
          <h2 id="type-cases-title">典型案例</h2>
          <ul class="type-case-list">
            <li v-for="item in cases" :key="item.id">
              <RouterLink
                :to="{
                  path: `/catalog/${item.landslideId}`,
                  query: { from: 'atlas', fromType: entry.type },
                }"
              >
                <span class="type-case-copy">
                  <strong>{{ caseTitle(item) }}</strong>
                  <span>编号 {{ item.landslideId.slice(3) }} · {{ item.subtitle }}</span>
                  <span v-if="!atlas.state.loading && !casesWithImage.has(item.id)"
                    >暂无公开影像</span
                  >
                </span>
                <ChevronRight :size="17" aria-hidden="true" />
              </RouterLink>
            </li>
          </ul>
        </section>

        <section class="type-landforms" aria-labelledby="type-landforms-title">
          <h2 id="type-landforms-title">地貌特征</h2>
          <dl>
            <div>
              <dt>堆积特征</dt>
              <dd>{{ entry.deposits }}</dd>
            </div>
            <div>
              <dt>流动特征</dt>
              <dd>{{ entry.flow }}</dd>
            </div>
          </dl>
        </section>

        <section class="type-confusion" aria-labelledby="type-confusion-title">
          <h2 id="type-confusion-title">易混淆类型</h2>
          <p>{{ entry.confusable }}</p>
        </section>
      </div>

      <details class="type-reference">
        <summary>
          <BookOpen :size="16" aria-hidden="true" />
          参考论文
          <ChevronDown :size="16" aria-hidden="true" />
        </summary>
        <div class="type-reference-content">
          <a :href="atlasContent.sourceDocument.url" target="_blank" rel="noopener noreferrer">
            {{ atlasContent.sourceDocument.title }} <ArrowUpRight :size="15" aria-hidden="true" />
          </a>
          <ul>
            <li v-for="citation in entry.citations" :key="citation.locator">
              {{ citation.locator }}
            </li>
          </ul>
          <p v-if="entry.catalogNote">{{ entry.catalogNote }}</p>
        </div>
      </details>
    </div>
  </section>
</template>

<style scoped>
.atlas-type-page {
  max-width: 1032px;
}
.type-page-links {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
  margin-block-end: var(--space-xs);
}
.type-page-back,
.type-page-compare {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2xs);
  min-height: 44px;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.type-page-back:hover,
.type-page-compare:hover,
.type-reference-content a:hover {
  color: var(--guide-color, var(--accent-hover));
}
.type-guide {
  --guide-section-space: clamp(18px, 3.1vw, 32px);
  --guide-padding: clamp(16px, 2.8vw, 24px);
}
.type-guide-nav {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--space-3xs);
  margin-block-start: var(--guide-section-space);
  border-block-end: 1px solid var(--line);
}
.type-guide-nav a {
  display: grid;
  place-items: center;
  min-height: 44px;
  border-block-end: 2px solid transparent;
  border-start-start-radius: 6px;
  border-start-end-radius: 6px;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  white-space: nowrap;
}
.type-guide-nav a:hover {
  background: var(--surface-raised);
  color: var(--text);
}
.type-guide-nav a.current {
  border-block-end-color: var(--guide-color);
  color: var(--guide-color);
  font-weight: 650;
}
.type-title {
  display: flex;
  align-items: baseline;
  gap: var(--space-sm);
}
.type-title h1 {
  margin: 0;
  font-size: clamp(1.75rem, 3.2vw, 2.5rem);
  font-weight: 650;
  line-height: 1.1;
}
.type-abbreviation {
  color: var(--guide-color);
  font-family: var(--font-outlier);
  font-size: var(--text-md);
}
.type-hero p {
  max-width: 46rem;
  margin: var(--space-sm) 0 0;
  color: var(--text-secondary);
  font-size: var(--text-md);
  line-height: 1.65;
}
.type-guide-grid {
  display: grid;
  gap: var(--guide-section-space);
  margin-block-start: var(--guide-section-space);
}
.type-guide-grid h2 {
  margin: 0 0 var(--space-sm);
  color: var(--text);
  font-size: var(--text-section);
  font-weight: 650;
}
.type-signals,
.type-confusion {
  padding: var(--guide-padding);
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
}
.type-signals {
  border-inline-start: 3px solid var(--guide-color);
}
.type-signals ul,
.type-case-list {
  margin: 0;
  padding: 0;
  list-style: none;
}
.type-signals ul {
  display: grid;
  gap: var(--space-sm);
  color: var(--text-secondary);
  line-height: 1.6;
}
.type-signals li {
  display: grid;
  grid-template-columns: 6px minmax(0, 1fr);
  gap: var(--space-xs);
}
.type-signals li::before {
  content: '';
  width: 6px;
  height: 6px;
  margin-top: 0.65em;
  border-radius: 50%;
  background: var(--guide-color);
}
.type-cases {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
}
.type-cases h2 {
  margin: 0;
  padding: var(--space-sm) var(--guide-padding);
}
.type-case-list li {
  border-top: 1px solid var(--line);
}
.type-case-list a {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--space-sm);
  min-height: 60px;
  padding: var(--space-xs) var(--guide-padding);
}
.type-case-list a:hover {
  background: var(--surface-raised);
}
.type-case-list a:focus-visible {
  outline-offset: -3px;
}
.type-case-list svg {
  color: var(--guide-color);
}
.type-case-copy {
  display: grid;
  gap: var(--space-3xs);
  line-height: 1.45;
}
.type-case-copy strong {
  color: var(--text);
  font-size: var(--text-sm);
  font-weight: 600;
}
.type-case-copy > span {
  color: var(--muted);
  font-size: var(--text-xs);
}
.type-landforms dl {
  display: grid;
  gap: var(--space-sm);
  margin: 0;
}
.type-landforms dl > div {
  padding-block: var(--space-sm) 0;
  border-block-start: 1px solid var(--line);
}
.type-landforms dt {
  font-weight: 600;
}
.type-landforms dd,
.type-confusion p {
  margin: var(--space-2xs) 0 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.7;
}
.type-reference {
  margin-block-start: var(--guide-section-space);
  border-block: 1px solid var(--line);
  color: var(--muted);
  font-size: var(--text-sm);
}
.type-reference summary {
  display: flex;
  min-height: 48px;
  align-items: center;
  gap: var(--space-xs);
  cursor: pointer;
  list-style: none;
}
.type-reference summary::-webkit-details-marker {
  display: none;
}
.type-reference summary svg:last-child {
  margin-inline-start: auto;
}
.type-reference[open] summary svg:last-child {
  transform: rotate(180deg);
}
.type-reference-content {
  padding-block-end: var(--space-md);
}
.type-reference-content a {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2xs);
  min-height: 44px;
  color: var(--text-secondary);
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
.type-reference-content ul {
  margin: 0 0 var(--space-xs);
  padding-inline-start: var(--space-md);
}
@media (min-width: 760px) {
  .type-guide-grid {
    grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
    grid-template-areas: 'signals cases' 'landforms confusion';
    align-items: start;
  }
  .type-signals {
    grid-area: signals;
  }
  .type-cases {
    grid-area: cases;
  }
  .type-landforms {
    grid-area: landforms;
  }
  .type-confusion {
    grid-area: confusion;
  }
}
</style>
