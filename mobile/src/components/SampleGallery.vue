<script setup lang="ts">
import { samples } from '../lib/samples'
import { typeInfo } from '../types'
defineProps<{ compact?: boolean }>()
</script>
<template>
  <section class="sample-gallery" aria-label="公开影像示例">
    <div class="section-heading">
      <div>
        <span class="eyebrow">从真实影像开始</span>
        <h2>认识一处火星滑坡</h2>
      </div>
      <RouterLink to="/atlas#samples">8 个示例 →</RouterLink>
    </div>
    <div class="sample-grid" :class="{ 'sample-strip': compact }">
      <RouterLink
        v-for="sample in samples"
        :key="sample.id"
        :to="`/samples/${sample.id}`"
        class="sample-card"
      >
        <img
          :src="sample.imageUrl"
          :alt="`滑坡 ${sample.sourceId} 的真实裁片`"
          width="224"
          height="224"
          :loading="compact ? 'eager' : 'lazy'"
        />
        <div>
          <span class="sample-kicker">{{ sample.type }} · {{ typeInfo[sample.type].name }}</span>
          <h3>滑坡 {{ sample.sourceId }}</h3>
          <p>查看影像与编目类型 <span>↗</span></p>
        </div>
      </RouterLink>
    </div>
  </section>
</template>
