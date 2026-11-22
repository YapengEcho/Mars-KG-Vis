<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { ArrowLeft, BookOpen, ChevronDown, ChevronRight } from '@lucide/vue'
import { atlasContent } from '../lib/atlas'
import { typeInfo } from '../types'

const route = useRoute()
const comparisons = atlasContent.comparisons
const comparison = computed(
  () => comparisons.find((entry) => entry.id === route.query.pair) ?? comparisons[0],
)
</script>
<template>
  <section class="content-view atlas-compare">
    <RouterLink to="/atlas" class="compare-back">
      <ArrowLeft :size="16" aria-hidden="true" />返回图鉴
    </RouterLink>
    <h1>滑坡类型对比</h1>

    <template v-if="comparison">
      <nav class="pair-selector" aria-label="对比组合">
        <RouterLink
          v-for="pair in comparisons"
          :key="pair.id"
          :to="{ path: '/atlas/compare', query: { pair: pair.id } }"
          :class="{ 'pair-selected': pair.id === comparison.id }"
          :aria-current-value="pair.id === comparison.id ? 'page' : 'false'"
        >
          {{ typeInfo[pair.left].name }}<span>与</span>{{ typeInfo[pair.right].name }}
        </RouterLink>
      </nav>

      <p class="compare-focus">{{ comparison.why }}</p>
      <table class="compare-matrix">
        <caption>
          {{
            comparison.title
          }}
        </caption>
        <thead>
          <tr>
            <th
              v-for="kind in [comparison.left, comparison.right]"
              :key="kind"
              scope="col"
              :style="{ '--compare-type-color': typeInfo[kind].color }"
            >
              <RouterLink :to="`/atlas/type/${kind}`">
                <span>{{ typeInfo[kind].name }}</span>
                <small>{{ kind }}</small>
                <ChevronRight :size="16" aria-hidden="true" />
              </RouterLink>
            </th>
          </tr>
        </thead>
        <tbody v-for="row in comparison.rows" :key="row.dimension">
          <tr>
            <th colspan="2" scope="rowgroup">{{ row.dimension }}</th>
          </tr>
          <tr>
            <td>{{ row.left }}</td>
            <td>{{ row.right }}</td>
          </tr>
        </tbody>
      </table>

      <details class="compare-notes" :key="comparison.id">
        <summary>
          <BookOpen :size="16" aria-hidden="true" />判读说明与参考论文
          <ChevronDown :size="16" aria-hidden="true" />
        </summary>
        <ul>
          <li v-for="limit in atlasContent.limits" :key="limit">{{ limit }}</li>
        </ul>
        <a :href="atlasContent.sourceDocument.url" target="_blank" rel="noopener noreferrer">
          {{ atlasContent.sourceDocument.title }}
        </a>
        <ul>
          <li v-for="reference in comparison.citations" :key="reference.locator">
            {{ reference.locator }}
          </li>
        </ul>
      </details>
    </template>
  </section>
</template>

<style scoped>
.atlas-compare {
  max-width: 1032px;
  --compare-gap: clamp(14px, 2.4vw, 24px);
  --compare-inset: clamp(12px, 2.3vw, 26px);
}
.compare-back {
  display: inline-flex;
  align-items: center;
  gap: var(--space-xs);
  min-height: 44px;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.compare-back:hover {
  color: var(--accent-hover);
}
.atlas-compare h1 {
  margin: var(--space-xs) 0 var(--compare-gap);
  font-size: clamp(1.75rem, 3.2vw, 2.5rem);
  font-weight: 650;
  line-height: 1.18;
}
.pair-selector {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs);
}
.pair-selector a {
  display: inline-flex;
  flex: 1 1 136px;
  align-items: center;
  justify-content: center;
  gap: var(--space-2xs);
  min-height: 44px;
  padding: var(--space-xs) var(--space-sm);
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  white-space: nowrap;
}
.pair-selector a:hover {
  border-color: var(--accent);
}
.pair-selector a.pair-selected {
  border-color: color-mix(in oklab, var(--accent) 65%, var(--line));
  background: var(--selected-bg);
  color: var(--accent-hover);
  font-weight: 650;
}
.pair-selector span {
  font-weight: 400;
}
.compare-focus {
  margin: var(--compare-gap) 0 var(--space-sm);
  padding: var(--space-sm) var(--compare-inset);
  border-inline-start: 2px solid var(--accent);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.65;
}
.compare-matrix {
  width: 100%;
  table-layout: fixed;
  border: 1px solid var(--line);
  border-radius: 12px;
  border-spacing: 0;
  background: var(--surface);
  text-align: left;
}
.compare-matrix caption {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}
.compare-matrix thead th {
  padding: 0;
  border-bottom: 1px solid var(--line);
  border-top: 3px solid var(--compare-type-color);
}
.compare-matrix thead th:first-child {
  border-top-left-radius: 11px;
}
.compare-matrix thead th:last-child {
  border-top-right-radius: 11px;
}
.compare-matrix th + th,
.compare-matrix td + td {
  border-inline-start: 1px solid var(--line);
}
.compare-matrix thead a {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  min-height: 52px;
  padding: var(--space-xs) var(--compare-inset);
  color: var(--compare-type-color);
  font-size: var(--text-md);
  font-weight: 650;
}
.compare-matrix thead a:hover {
  background: var(--surface-raised);
}
.compare-matrix thead a:focus-visible {
  outline-offset: -3px;
}
.compare-matrix thead small {
  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: 500;
}
.compare-matrix thead svg {
  flex: none;
  margin-inline-start: auto;
}
.compare-matrix tbody th {
  padding: var(--space-sm) var(--compare-inset) var(--space-3xs);
  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: 600;
}
.compare-matrix tbody + tbody th {
  border-block-start: 1px solid var(--line);
}
.compare-matrix td {
  padding: var(--space-2xs) var(--compare-inset) var(--space-sm);
  vertical-align: top;
  color: var(--text);
  font-size: var(--text-sm);
  line-height: 1.6;
  overflow-wrap: anywhere;
}
.compare-notes {
  margin-block-start: var(--compare-gap);
  border-block: 1px solid var(--line);
  color: var(--muted);
  font-size: var(--text-sm);
  line-height: 1.65;
}
.compare-notes summary {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  min-height: 48px;
  cursor: pointer;
  list-style: none;
}
.compare-notes summary::-webkit-details-marker {
  display: none;
}
.compare-notes summary svg:last-child {
  margin-inline-start: auto;
}
.compare-notes[open] summary svg:last-child {
  transform: rotate(180deg);
}
.compare-notes ul {
  display: grid;
  gap: var(--space-2xs);
  margin: 0 0 var(--space-md);
  padding-inline-start: var(--space-lg);
}
.compare-notes a {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  color: var(--accent-hover);
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
@media (min-width: 720px) {
  .pair-selector a {
    flex: 0 0 auto;
    padding-inline: var(--space-lg);
  }
  .compare-matrix thead a {
    font-size: var(--text-lg);
  }
  .compare-matrix td {
    font-size: var(--text-md);
  }
}
</style>
