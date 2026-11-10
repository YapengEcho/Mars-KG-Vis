<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowRight, ArrowUpRight, Search } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import SampleGallery from '../components/SampleGallery.vue'
import { atlas, byType } from '../stores/atlas'
import { atlasContent, casesOfType } from '../lib/atlas'
import { findSlide, filterSlides } from '../lib/catalog'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'
const query = ref('')
const type = ref<SlideType | ''>('')
const counts = computed(() => byType(atlas.state.catalog.records))
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
const cases = computed(() => atlasContent.cases)
function hasRecord(landslideId: string) {
  return Boolean(findSlide(atlas.state.catalog.records, landslideId))
}
function openRecord(landslideId: string) {
  const record = findSlide(atlas.state.catalog.records, landslideId)
  if (record) atlas.select(record.id)
}
</script>
<template>
  <section class="content-view">
    <div class="page-heading">
      <span class="eyebrow">THE LANDSLIDE ATLAS</span>
      <h1>滑坡类型与目录</h1>
      <p>先看四类的定义与区别，再按目录查记录。</p>
    </div>
    <div class="type-grid">
      <RouterLink
        v-for="t in slideTypes"
        :key="t"
        :to="`/atlas/type/${t}`"
        class="type-tile"
        :style="{ '--type-color': typeInfo[t].color }"
      >
        <div>
          <span class="type-code">{{ t }}</span
          ><ArrowUpRight :size="18" />
        </div>
        <h2>{{ typeInfo[t].name }}</h2>
        <span class="type-english">{{ typeInfo[t].english }}</span>
        <footer>{{ counts[t] ?? 0 }} 条记录 · {{ casesOfType(t).length }} 例</footer>
      </RouterLink>
    </div>
    <p class="type-note">
      {{ type ? typeInfo[type].note : '类别沿用目录标签；点位详情中显示对应来源。' }}
    </p>
    <SampleGallery id="samples" />

    <div class="atlas-entry">
      <div>
        <span class="eyebrow">COMPARE</span>
        <h2>最容易混的两组怎么分</h2>
        <p class="muted">SL 与 SP、SP 与 CL，拆成可以逐条对照的维度。</p>
      </div>
      <RouterLink class="secondary-button" to="/atlas/compare"
        >两类对比 <ArrowRight :size="16"
      /></RouterLink>
    </div>

    <div class="section-heading">
      <h2>
        12 个案例 <span>{{ cases.length }}</span>
      </h2>
      <RouterLink to="/atlas/compare">判读限制</RouterLink>
    </div>
    <div class="case-grid">
      <article v-for="item in cases" :key="item.id" class="case-card">
        <span class="eyebrow">{{ item.type }} · {{ item.tile }}</span>
        <h3>
          <RouterLink :to="`/atlas/case/${item.id}`">{{ item.title }}</RouterLink>
        </h3>
        <p class="muted">{{ item.subtitle }} · {{ item.previewScale }}</p>
        <div class="tag-row">
          <span v-for="tag in item.tags" :key="tag" class="data-badge">{{ tag }}</span>
        </div>
        <div class="case-actions">
          <RouterLink class="text-button" :to="`/atlas/case/${item.id}`"
            >案例页 <ArrowRight :size="15"
          /></RouterLink>
          <button
            class="text-button"
            :disabled="!hasRecord(item.landslideId)"
            @click="openRecord(item.landslideId)"
          >
            {{ hasRecord(item.landslideId) ? '目录记录' : '示例目录中无记录' }}
          </button>
        </div>
      </article>
    </div>

    <div class="section-heading">
      <h2>
        {{ type ? `${typeInfo[type].name}目录` : '全部目录' }} <span>{{ matches.length }}</span>
      </h2>
      <button v-if="type" class="text-button" @click="type = ''">清除筛选</button>
    </div>
    <div class="type-filters">
      <button :class="{ active: type === '' }" @click="type = ''">全部</button>
      <button
        v-for="t in slideTypes"
        :key="t"
        :class="{ active: type === t }"
        @click="type = type === t ? '' : t"
      >
        <i :style="{ background: typeInfo[t].color }"></i>{{ t }}
      </button>
    </div>
    <div class="search-box">
      <Search :size="18" /><input
        v-model="query"
        aria-label="搜索目录"
        placeholder="输入编号、名称或说明"
      />
    </div>
    <div class="record-list">
      <SlideCard v-for="slide in matches" :key="slide.id" :slide="slide" />
      <p v-if="!matches.length" class="empty-inline">没有匹配记录。</p>
    </div>
  </section>
</template>
