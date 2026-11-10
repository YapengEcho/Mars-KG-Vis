<script setup lang="ts">
import { computed } from 'vue'
import ZoomableImage from '../components/ZoomableImage.vue'
import { sampleById } from '../lib/samples'
import { typeInfo } from '../types'
import { publicCatalog } from '../lib/publicCatalog'
import { coordinateStatus } from '../lib/coordinateStatus'
const props = defineProps<{ id: string }>()
const sample = computed(() => sampleById(props.id))
const coordinates = coordinateStatus(publicCatalog)
</script>
<template>
  <section class="content-view sample-view">
    <RouterLink to="/atlas#samples" class="text-button back-link">← 返回目录</RouterLink>
    <template v-if="sample">
      <div class="page-heading">
        <h1>滑坡 {{ sample.sourceId }}</h1>
        <p>编目类型：{{ typeInfo[sample.type].name }}（{{ sample.type }}）</p>
      </div>
      <div class="sample-media">
        <ZoomableImage :src="sample.imageUrl" :alt="`滑坡 ${sample.sourceId}`" />
      </div>
      <p class="sample-caption">224 × 224 像素</p>
      <div class="sample-actions">
        <RouterLink :to="`/recognize?sample=${sample.id}`" class="primary-button"
          >识别这张影像</RouterLink
        ><a :href="sample.imageUrl" :download="`${sample.id}.png`" class="secondary-button"
          >下载影像</a
        >
      </div>
      <RouterLink class="atlas-entry" :to="`/atlas/type/${sample.type}`"
        ><div>
          <h2>了解{{ typeInfo[sample.type].name }}</h2>
        </div>
        <span>→</span></RouterLink
      >
      <details class="quiet-details">
        <summary>来源与使用说明</summary>
        <p class="muted">
          图片来自已有训练裁片；试用结果不作为独立测试成绩。类别沿用原编目，单张影像不能代表这一类的所有形态。
        </p>
        <p class="muted">当前公开8张影像。{{ coordinates.note }}</p>
        <a :href="sample.source.url" target="_blank" rel="noopener noreferrer" class="text-button"
          >编目来源：{{ sample.source.title }} ↗</a
        >
      </details>
    </template>
    <p v-else class="notice">没有找到这张影像。</p>
  </section>
</template>
