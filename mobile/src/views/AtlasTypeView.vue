<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeft, ArrowRight, BookOpen } from '@lucide/vue'
import { atlas } from '../stores/atlas'
import { atlasContent, caseRecord, casesOfType, typeEntry } from '../lib/atlas'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'

const props = defineProps<{ type: string }>()
const type = computed(() => String(props.type ?? '').toUpperCase() as SlideType)
const valid = computed(() => slideTypes.includes(type.value))
const entry = computed(() => (valid.value ? typeEntry(type.value) : undefined))
const cases = computed(() => (valid.value ? casesOfType(type.value) : []))
const others = computed(() => slideTypes.filter((t) => t !== type.value))
function recordOf(caseId: string) {
  const item = cases.value.find((c) => c.id === caseId)
  return item ? caseRecord(item, atlas.state.catalog.records) : undefined
}
function openRecord(caseId: string) {
  const record = recordOf(caseId)
  if (record) atlas.select(record.id)
}
</script>
<template>
  <section class="content-view">
    <RouterLink to="/atlas" class="text-button back-link"
      ><ArrowLeft :size="16" />返回图鉴</RouterLink
    >
    <p v-if="!valid" class="notice">未知的滑坡类别：{{ props.type }}</p>
    <template v-else-if="entry">
      <div class="page-heading atlas-heading">
        <span class="eyebrow">{{ entry.type }} · {{ entry.english }}</span>
        <h1>
          <span class="atlas-type-code" :style="{ color: typeInfo[entry.type].color }">{{
            entry.type
          }}</span>
          {{ entry.name }}
        </h1>
        <p>{{ entry.quote }}</p>
      </div>

      <dl class="atlas-facts">
        <div>
          <dt>后运动堆积特征</dt>
          <dd>{{ entry.deposits }}</dd>
        </div>
        <div>
          <dt>流动特征</dt>
          <dd>{{ entry.flow }}</dd>
        </div>
        <div>
          <dt>怎么认</dt>
          <dd>
            <ol class="atlas-points">
              <li v-for="point in entry.recognize" :key="point">{{ point }}</li>
            </ol>
          </dd>
        </div>
        <div>
          <dt>最容易混</dt>
          <dd>{{ entry.confusable }}</dd>
        </div>
      </dl>

      <div class="section-heading">
        <h2>{{ entry.name }}案例 <span>{{ cases.length }}</span></h2>
        <RouterLink to="/atlas/compare">两类对比</RouterLink>
      </div>
      <div class="case-grid">
        <article v-for="item in cases" :key="item.id" class="case-card">
          <span class="eyebrow">{{ item.tile }} · {{ item.previewScale }}</span>
          <h3>
            <RouterLink :to="`/atlas/case/${item.id}`">{{ item.title }}</RouterLink>
          </h3>
          <p class="muted">{{ item.subtitle }}</p>
          <p>{{ item.observations[0] }}</p>
          <div class="tag-row">
            <span v-for="tag in item.tags" :key="tag" class="data-badge">{{ tag }}</span>
          </div>
          <div class="case-actions">
            <RouterLink class="secondary-button" :to="`/atlas/case/${item.id}`"
              >案例页 <ArrowRight :size="15"
            /></RouterLink>
            <button class="text-button" :disabled="!recordOf(item.id)" @click="openRecord(item.id)">
              {{ recordOf(item.id) ? '查看目录记录' : '示例目录中无此记录' }}
            </button>
          </div>
        </article>
      </div>

      <div class="section-heading"><h2>引用与出处</h2></div>
      <ul class="citation-list">
        <li v-for="c in entry.citations" :key="c.locator">
          <BookOpen :size="15" />
          <span>{{ c.locator }}</span>
        </li>
      </ul>
      <p class="page-footnote">
        引文与页码取自《{{ atlasContent.sourceDocument.title }}》（{{
          atlasContent.sourceDocument.note
        }}，{{ atlasContent.sourceDocument.url }}）。图鉴只引用专家编目的定义，不替代现场判读。
      </p>

      <div class="section-heading"><h2>其它类别</h2></div>
      <div class="type-filters">
        <RouterLink
          v-for="t in others"
          :key="t"
          :to="`/atlas/type/${t}`"
          :class="['type-pill', `type-${t}`]"
        >
          {{ t }} {{ typeInfo[t].name }}
        </RouterLink>
      </div>
    </template>
  </section>
</template>
