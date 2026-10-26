<script setup lang="ts">
import { computed, ref } from 'vue'
import { Search, ArrowUpRight } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas, byType } from '../stores/atlas'
import { slideTypes, typeInfo } from '../types'
import type { SlideType } from '../types'
import { filterSlides } from '../lib/catalog'
const query = ref('')
const type = ref<SlideType | ''>('')
const counts = computed(() => byType(atlas.state.catalog.records))
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
</script>
<template>
  <section class="content-view">
    <div class="page-heading">
      <span class="eyebrow">THE LANDSLIDE ATLAS</span>
      <h1>滑坡类型与目录</h1>
      <p>从四类目录出发，查找点位与案例。</p>
    </div>
    <div class="type-grid">
      <button
        v-for="t in slideTypes"
        :key="t"
        class="type-tile"
        :class="{ selected: type === t }"
        :style="{ '--type-color': typeInfo[t].color }"
        :aria-pressed="type === t"
        @click="type = type === t ? '' : t"
      >
        <div>
          <span class="type-code">{{ t }}</span
          ><ArrowUpRight :size="18" />
        </div>
        <h2>{{ typeInfo[t].name }}</h2>
        <span class="type-english">{{ typeInfo[t].english }}</span>
        <footer>{{ counts[t] ?? 0 }} 条记录</footer>
      </button>
    </div>
    <p class="type-note">
      {{ type ? typeInfo[type].note : '类别沿用目录标签；点位详情中显示对应来源。' }}
    </p>
    <div class="section-heading">
      <h2>
        {{ type ? `${typeInfo[type].name}目录` : '全部目录' }} <span>{{ matches.length }}</span>
      </h2>
      <button v-if="type" class="text-button" @click="type = ''">清除筛选</button>
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
