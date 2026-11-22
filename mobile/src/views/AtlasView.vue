<script setup lang="ts">
import { ArrowUpRight, ChevronRight } from '@lucide/vue'
import { atlasContent, casesOfType, caseTitle } from '../lib/atlas'
import { slideTypes, typeInfo } from '../types'

const groups = slideTypes.map((type) => ({
  code: type,
  info: typeInfo[type],
  cases: casesOfType(type),
}))
</script>

<template>
  <section class="content-view curated-atlas">
    <header class="atlas-heading">
      <div>
        <h1>滑坡图鉴</h1>
        <p>{{ atlasContent.cases.length }} 个案例，覆盖四种滑坡类型</p>
      </div>
      <RouterLink class="atlas-compare-link" to="/atlas/compare">
        类型对比 <ArrowUpRight :size="16" aria-hidden="true" />
      </RouterLink>
    </header>

    <div id="samples" class="case-groups">
      <section
        v-for="group in groups"
        :key="group.code"
        class="case-group"
        :aria-labelledby="`case-group-${group.code}`"
        :style="{ '--case-type-color': group.info.color }"
      >
        <h2 :id="`case-group-${group.code}`">
          <RouterLink class="case-guide-link" :to="`/atlas/type/${group.code}`">
            {{ group.info.name }} <span>{{ group.code }}</span>
            <ChevronRight :size="16" aria-hidden="true" />
          </RouterLink>
        </h2>
        <ul class="atlas-case-list">
          <li v-for="item in group.cases" :key="item.id">
            <RouterLink :to="{ path: `/catalog/${item.landslideId}`, query: { from: 'atlas' } }">
              <span class="case-identity">
                <span class="case-title">{{ caseTitle(item) }}</span>
                <span class="case-number">编号 {{ item.landslideId.slice(3) }}</span>
              </span>
              <ChevronRight :size="17" aria-hidden="true" />
            </RouterLink>
          </li>
        </ul>
      </section>
    </div>
  </section>
</template>

<style scoped>
.curated-atlas {
  width: min(100%, 1032px);
}

.atlas-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  justify-content: space-between;
  gap: var(--space-sm) var(--space-md);
  margin-block-end: var(--space-lg);
}

.atlas-heading h1 {
  margin: 0;
  font-size: clamp(1.75rem, 3.2vw, 2.5rem);
  font-weight: 650;
  line-height: 1.18;
}

.atlas-heading p {
  margin: var(--space-2xs) 0 0;
  color: var(--muted);
  font-size: var(--text-sm);
}

.atlas-compare-link {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  gap: var(--space-xs);
  padding-inline: var(--space-sm);
  border: 1px solid color-mix(in oklab, var(--accent) 45%, var(--line));
  border-radius: 8px;
  background: var(--selected-bg);
  color: var(--accent-hover);
  font-size: var(--text-sm);
  white-space: nowrap;
}

.atlas-compare-link:hover {
  border-color: var(--accent);
}

.case-groups {
  display: grid;
  gap: clamp(14px, 2.8vw, 24px);
  scroll-margin-top: 16px;
}

.case-group {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
}

.case-group > h2 {
  max-width: none;
  margin: 0;
  border-block-end: 1px solid var(--line);
  color: var(--case-type-color);
  font-size: var(--text-lg);
  font-weight: 600;
}

.case-guide-link {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  min-height: 50px;
  padding: var(--space-xs) var(--space-md);
  border-start-start-radius: 11px;
  border-start-end-radius: 11px;
}

.case-guide-link:hover {
  background: var(--surface-raised);
}

.case-guide-link:focus-visible {
  outline-offset: -2px;
}

.case-guide-link svg {
  margin-inline-start: auto;
}

.case-group > h2 span {
  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: 500;
}

.atlas-case-list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.atlas-case-list li + li {
  border-block-start: 1px solid var(--line);
}

.atlas-case-list a {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
  min-height: 60px;
  padding: var(--space-xs) var(--space-md);
  color: var(--text);
  text-decoration: none;
}

.atlas-case-list a:hover,
.atlas-case-list a:focus-visible {
  background: var(--surface-raised);
}

.atlas-case-list li:last-child a {
  border-end-start-radius: 11px;
  border-end-end-radius: 11px;
}

.atlas-case-list a:focus-visible {
  outline-offset: -2px;
}

.atlas-case-list svg {
  flex: none;
  color: var(--muted);
}

.case-identity {
  display: grid;
  gap: var(--space-3xs);
  min-width: 0;
}

.case-title {
  font-weight: 500;
}

.case-number {
  color: var(--muted);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

@media (min-width: 720px) {
  .case-groups {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
