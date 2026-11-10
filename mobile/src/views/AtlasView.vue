<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowRight, Search } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas } from '../stores/atlas'
import { atlasContent } from '../lib/atlas'
import { filterSlides } from '../lib/catalog'
import { samples } from '../lib/samples'
import { slideTypes, typeInfo, type SlideType } from '../types'
const query = ref('')
const type = ref<SlideType | ''>('')
const matches = computed(() => filterSlides(atlas.state.catalog.records, query.value, type.value))
</script>
<template>
  <section class="content-view atlas-view">
    <div class="page-heading"><h1>滑坡图鉴</h1></div>
    <div class="type-grid">
      <RouterLink
        v-for="t in slideTypes"
        :key="t"
        :to="`/atlas/type/${t}`"
        class="type-tile visual-type"
        :style="{ '--type-color': typeInfo[t].color }"
      >
        <img :src="samples.find((s) => s.type === t)?.imageUrl" :alt="typeInfo[t].name" />
        <div>
          <span class="type-code">{{ t }}</span>
          <h2>{{ typeInfo[t].name }}</h2>
          <ArrowRight :size="16" />
        </div>
      </RouterLink>
    </div>
    <RouterLink class="compact-compare" to="/atlas/compare"
      >易混类型对比 <ArrowRight :size="16"
    /></RouterLink>
    <section id="samples" class="atlas-catalog">
      <div class="section-heading">
        <h2>
          影像目录 <span>{{ matches.length }}</span>
        </h2>
      </div>
      <div class="search-box">
        <Search :size="18" /><input v-model="query" aria-label="搜索目录" placeholder="搜索编号" />
      </div>
      <div class="type-filters" role="group" aria-label="目录类型筛选">
        <button :class="{ active: !type }" :aria-pressed="!type" @click="type = ''">全部</button
        ><button
          v-for="t in slideTypes"
          :key="t"
          :class="{ active: type === t }"
          :aria-pressed="type === t"
          @click="type = type === t ? '' : t"
        >
          {{ typeInfo[t].name }}
        </button>
      </div>
      <div class="record-list">
        <SlideCard v-for="slide in matches" :key="slide.id" :slide="slide" />
        <p v-if="!matches.length" class="empty-inline">没有匹配记录。</p>
      </div>
    </section>
    <details class="quiet-details case-reading">
      <summary>延伸阅读 · {{ atlasContent.cases.length }} 个案例</summary>
      <div class="case-reading-list">
        <RouterLink v-for="item in atlasContent.cases" :key="item.id" :to="`/atlas/case/${item.id}`"
          ><span>{{ item.title }}</span
          ><ArrowRight :size="15"
        /></RouterLink>
      </div>
    </details>
  </section>
</template>
