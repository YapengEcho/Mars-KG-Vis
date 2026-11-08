<script setup lang="ts">
import { computed } from 'vue'
import { ArrowLeft, BookOpen } from '@lucide/vue'
import { atlasContent, casesOfType } from '../lib/atlas'
import { typeInfo } from '../types'
const comparisons = computed(() => atlasContent.comparisons)
const caseCount = (type: 'SL' | 'SP' | 'FL' | 'CL') => casesOfType(type).length
</script>
<template>
  <section class="content-view">
    <RouterLink to="/atlas" class="text-button back-link"
      ><ArrowLeft :size="16" />返回图鉴</RouterLink
    >
    <div class="page-heading">
      <span class="eyebrow">TYPE COMPARISON</span>
      <h1>两类对比</h1>
      <p>挑最容易混的两组，把差别拆成可以直接查的维度。</p>
    </div>

    <article v-for="comparison in comparisons" :key="comparison.id" class="compare-block">
      <header>
        <h2>{{ comparison.title }}</h2>
        <p class="muted">{{ comparison.why }}</p>
      </header>
      <table class="compare-table">
        <thead>
          <tr>
            <th scope="col">维度</th>
            <th scope="col">
              <RouterLink :to="`/atlas/type/${comparison.left}`">
                {{ comparison.left }} {{ typeInfo[comparison.left].name }}
              </RouterLink>
              <span class="muted">{{ caseCount(comparison.left) }} 例</span>
            </th>
            <th scope="col">
              <RouterLink :to="`/atlas/type/${comparison.right}`">
                {{ comparison.right }} {{ typeInfo[comparison.right].name }}
              </RouterLink>
              <span class="muted">{{ caseCount(comparison.right) }} 例</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in comparison.rows" :key="row.dimension">
            <th scope="row">{{ row.dimension }}</th>
            <td>{{ row.left }}</td>
            <td>{{ row.right }}</td>
          </tr>
        </tbody>
      </table>
      <ul class="citation-list">
        <li v-for="c in comparison.citations" :key="c.locator">
          <BookOpen :size="15" /><span>{{ c.locator }}</span>
        </li>
      </ul>
    </article>

    <div class="section-heading"><h2>判读限制</h2></div>
    <ul class="limit-list">
      <li v-for="limit in atlasContent.limits" :key="limit">{{ limit }}</li>
    </ul>
    <p class="page-footnote">{{ atlasContent.dataNote }}</p>
  </section>
</template>
