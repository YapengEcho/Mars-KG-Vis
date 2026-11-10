<script setup lang="ts">
import { computed } from 'vue'
import ZoomableImage from '../components/ZoomableImage.vue'
import { sampleById } from '../lib/samples'
import { typeInfo } from '../types'
const props = defineProps<{ id: string }>()
const sample = computed(() => sampleById(props.id))
</script>
<template>
  <section class="content-view sample-view">
    <RouterLink to="/atlas#samples" class="text-button back-link">← 返回影像示例</RouterLink>
    <template v-if="sample">
      <div class="page-heading">
        <span class="eyebrow">真实影像 · 公开示例</span>
        <h1>滑坡 {{ sample.sourceId }}</h1>
        <p>编目类型：{{ typeInfo[sample.type].name }}（{{ sample.type }}）</p>
      </div>
      <div class="sample-media">
        <ZoomableImage :src="sample.imageUrl" :alt="`滑坡 ${sample.sourceId}`" />
      </div>
      <p class="sample-caption">224 × 224 像素 · 灰度裁片 · 可缩放查看</p>
      <div class="sample-actions">
        <RouterLink :to="`/recognize?sample=${sample.id}`" class="primary-button"
          >用这张图试分类</RouterLink
        ><a :href="sample.imageUrl" :download="`${sample.id}.png`" class="secondary-button"
          >下载示例</a
        >
      </div>
      <RouterLink class="atlas-entry" :to="`/atlas/type/${sample.type}`"
        ><div>
          <h2>{{ typeInfo[sample.type].name }}有什么特征？</h2>
          <p>查看图鉴中的定义、观察要点与易混类型。</p>
        </div>
        <span>→</span></RouterLink
      >
      <p class="muted">
        图片来自已有训练裁片；试用结果用于体验流程，不作为独立测试成绩。类别沿用原编目，示例不能代表这一类的所有形态。
      </p>
      <p class="muted">
        这 8 张是允许公开的部分示例。坐标约定尚待核准，因此未作为正式点位叠加到火星球。
      </p>
      <a :href="sample.source.url" target="_blank" rel="noopener noreferrer" class="text-button"
        >编目来源：{{ sample.source.title }} ↗</a
      >
    </template>
    <p v-else class="notice">没有找到这个公开示例。</p>
  </section>
</template>
