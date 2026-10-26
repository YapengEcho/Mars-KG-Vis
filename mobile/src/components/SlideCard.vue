<script setup lang="ts">
import { ArrowUpRight, Bookmark } from '@lucide/vue'
import type { Slide } from '../types'
import { typeInfo, coordinateLabel } from '../types'
import { atlas } from '../stores/atlas'
defineProps<{ slide: Slide; compact?: boolean }>()
</script>
<template>
  <article class="slide-card" :class="{ compact }">
    <button
      class="slide-card-main"
      @click="atlas.select(slide.id)"
      :aria-label="`查看${slide.name}详情`"
    >
      <div class="terrain-swatch" :style="{ '--swatch': typeInfo[slide.type].color }">
        <span>{{ slide.type }}</span>
      </div>
      <div class="slide-card-copy">
        <span class="eyebrow">{{ slide.id }}</span>
        <h3>{{ slide.name }}</h3>
        <p>{{ coordinateLabel(slide.longitude, slide.latitude) }}</p>
      </div>
      <ArrowUpRight :size="18" class="muted" />
    </button>
    <button
      class="icon-button save-icon"
      @click="atlas.toggleSaved(slide.id)"
      :aria-label="atlas.isSaved(slide.id) ? '取消收藏' : '收藏'"
      :aria-pressed="atlas.isSaved(slide.id)"
    >
      <Bookmark :size="17" :fill="atlas.isSaved(slide.id) ? 'currentColor' : 'none'" />
    </button>
  </article>
</template>
