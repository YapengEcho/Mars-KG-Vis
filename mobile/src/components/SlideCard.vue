<script setup lang="ts">
import { computed, ref } from 'vue'
import { Bookmark, ChevronRight, Crosshair } from '@lucide/vue'
import { useRoute } from 'vue-router'
import { publicAssetUrl } from '../lib/publicAssets'
import type { Slide } from '../types'
import { coordinateLabel, typeInfo } from '../types'
import { atlas } from '../stores/atlas'
defineProps<{ slide: Slide; compact?: boolean; highlighted?: boolean }>()
defineEmits<{ locate: [id: string] }>()
const unavailableImage = ref<string | null>(null)
const route = useRoute()
const back = computed(() =>
  route.path === '/mine' ? 'mine' : route.path === '/chat' ? 'chat' : undefined,
)
</script>
<template>
  <article
    class="slide-card"
    :class="{ compact, 'is-highlighted': highlighted }"
    :data-slide-id="slide.id"
  >
    <RouterLink
      class="slide-card-main"
      :to="{ name: 'catalog-record', params: { id: slide.id }, query: { from: back } }"
      :aria-label="`查看${slide.name}详情`"
    >
      <img
        v-if="
          !compact && slide.imageUrl?.startsWith('/samples/') && unavailableImage !== slide.imageUrl
        "
        class="slide-thumbnail"
        :src="publicAssetUrl(slide.imageUrl)"
        :alt="slide.name"
        width="224"
        height="224"
        loading="lazy"
        decoding="async"
        referrerpolicy="no-referrer"
        @error="unavailableImage = slide.imageUrl"
      />
      <div class="slide-card-copy">
        <h3>{{ slide.name }}</h3>
        <p class="card-facts">
          <span :style="{ color: typeInfo[slide.type].color }"
            >{{ typeInfo[slide.type].name }} {{ slide.type }}</span
          >
          <span v-if="compact && slide.areaKm2 !== null">{{ slide.areaKm2.toFixed(2) }} km²</span>
        </p>
        <p v-if="compact" class="card-coordinates">
          {{ coordinateLabel(slide.longitude, slide.latitude) }}
        </p>
      </div>
      <ChevronRight :size="17" class="card-detail-arrow" />
    </RouterLink>
    <button
      v-if="compact"
      type="button"
      class="map-shortcut"
      :aria-label="`定位${slide.name}`"
      @click="$emit('locate', slide.id)"
    >
      <Crosshair :size="16" /> 定位
    </button>
    <button
      v-else
      class="icon-button save-icon"
      @click="atlas.toggleSaved(slide.id)"
      :aria-label="atlas.isSaved(slide.id) ? '取消收藏' : '收藏'"
      :aria-pressed="atlas.isSaved(slide.id)"
    >
      <Bookmark :size="17" :fill="atlas.isSaved(slide.id) ? 'currentColor' : 'none'" />
    </button>
  </article>
</template>

<style scoped>
.slide-card-main {
  color: inherit;
  text-decoration: none;
}
.compact .slide-card-main {
  display: flex;
  align-items: center;
  min-height: 96px;
  padding: 12px 72px 12px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.compact .slide-card-main:hover,
.compact.is-highlighted .slide-card-main {
  border-color: var(--accent);
}
.compact.is-highlighted {
  outline: none;
}
.compact .slide-card-copy {
  display: block;
  padding: 0;
}
.compact .slide-card-copy .card-facts {
  display: flex;
  flex-wrap: wrap;
  gap: 3px 9px;
  padding: 0;
  margin: 2px 0;
  font-variant-numeric: tabular-nums;
}
.compact .slide-card-copy .card-coordinates {
  display: block;
  padding: 0;
  margin: 2px 0 0;
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}
.map-shortcut {
  position: absolute;
  inset-inline-end: 8px;
  bottom: 8px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding-inline: 6px;
  min-height: 44px;
  font-size: 0.78rem;
  color: var(--accent);
  border: 1px solid var(--line);
  border-radius: 5px;
  background: var(--surface);
  cursor: pointer;
}
.map-shortcut:hover {
  border-color: var(--accent);
}
</style>
