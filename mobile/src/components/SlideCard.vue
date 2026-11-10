<script setup lang="ts">
import { ArrowUpRight, Bookmark } from '@lucide/vue'
import type { Slide } from '../types'
import { typeInfo } from '../types'
import { atlas } from '../stores/atlas'
defineProps<{ slide: Slide; compact?: boolean; highlighted?: boolean }>()
</script>
<template>
  <article
    class="slide-card"
    :class="{ compact, 'is-highlighted': highlighted }"
    :data-slide-id="slide.id"
  >
    <button
      class="slide-card-main"
      @click="atlas.select(slide.id)"
      :aria-label="`查看${slide.name}详情`"
    >
      <img
        v-if="slide.imageUrl"
        class="slide-thumbnail"
        :src="slide.imageUrl"
        :alt="slide.name"
        width="224"
        height="224"
      />
      <div v-else class="terrain-swatch" :style="{ '--swatch': typeInfo[slide.type].color }">
        <span>{{ slide.type }}</span>
      </div>
      <div class="slide-card-copy">
        <h3>{{ slide.name }}</h3>
        <p>
          <i :style="{ background: typeInfo[slide.type].color }"></i>{{ slide.type }} ·
          {{ typeInfo[slide.type].name }}
        </p>
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
