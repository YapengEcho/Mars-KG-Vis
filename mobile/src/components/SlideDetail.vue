<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { Bookmark, Crosshair, ExternalLink, X } from '@lucide/vue'
import { useRouter } from 'vue-router'
import { atlas } from '../stores/atlas'
import { coordinateLabel, typeInfo } from '../types'
const router = useRouter()
const slide = atlas.selected
const dialog = ref<HTMLDialogElement>()
const imageFailed = ref(false)
const image = computed(() => !imageFailed.value && slide.value?.imageUrl)
let previousFocus: HTMLElement | null = null
watch(slide, async (value) => {
  imageFailed.value = false
  if (value) {
    previousFocus = document.activeElement as HTMLElement
    await nextTick()
    dialog.value?.showModal()
    document.body.style.overflow = 'hidden'
  } else {
    dialog.value?.close()
    document.body.style.overflow = ''
    previousFocus?.focus()
  }
})
onBeforeUnmount(() => {
  document.body.style.overflow = ''
})
function dismissBackdrop(event: MouseEvent) {
  if (event.target === dialog.value) atlas.dismiss()
}
async function locate() {
  const id = slide.value?.id
  if (!id) return
  atlas.dismiss()
  await router.push({ path: '/', query: { focus: id, visit: String(Date.now()) } })
}
</script>
<template>
  <dialog
    ref="dialog"
    class="detail-dialog"
    aria-labelledby="detail-title"
    @cancel.prevent="atlas.dismiss()"
    @click="dismissBackdrop"
  >
    <section v-if="slide" class="detail-sheet">
      <div class="sheet-handle"></div>
      <header class="detail-top">
        <span class="eyebrow">LANDSLIDE / {{ slide.id }}</span
        ><button class="icon-button" aria-label="关闭详情" @click="atlas.dismiss()">
          <X :size="21" />
        </button>
      </header>
      <div class="detail-media">
        <img v-if="image" :src="String(image)" :alt="slide.name" @error="imageFailed = true" />
        <div v-else class="contour-art">
          <Crosshair :size="33" :stroke-width="1" /><span>{{
            slide.isDemo ? '示例点位 · 非真实滑坡记录' : '暂无展示影像'
          }}</span>
        </div>
      </div>
      <span class="type-badge" :style="{ color: typeInfo[slide.type].color }"
        >{{ slide.type }} / {{ typeInfo[slide.type].name }}</span
      >
      <h2 id="detail-title">{{ slide.name }}</h2>
      <p class="detail-description">{{ slide.summary }}</p>
      <dl class="detail-grid">
        <div>
          <dt>火星经纬度</dt>
          <dd>{{ coordinateLabel(slide.longitude, slide.latitude) }}</dd>
        </div>
        <div>
          <dt>面积</dt>
          <dd>{{ slide.areaKm2 === null ? '暂无可靠记录' : `${slide.areaKm2.toFixed(2)} km²` }}</dd>
        </div>
        <div>
          <dt>原始编号</dt>
          <dd>{{ slide.sourceId ?? '示例记录' }}</dd>
        </div>
        <div>
          <dt>资料来源</dt>
          <dd>
            <a
              v-if="slide.source.url?.startsWith('https://')"
              :href="slide.source.url"
              target="_blank"
              rel="noopener noreferrer"
              >{{ slide.source.title }} <ExternalLink :size="12" /></a
            ><span v-else>{{ slide.source.title }}</span>
          </dd>
        </div>
      </dl>
      <div class="detail-actions">
        <button class="primary-button" @click="locate"><Crosshair :size="18" />定位到火星球</button
        ><button
          class="secondary-button"
          @click="atlas.toggleSaved(slide.id)"
          :aria-pressed="atlas.isSaved(slide.id)"
        >
          <Bookmark :size="17" :fill="atlas.isSaved(slide.id) ? 'currentColor' : 'none'" />{{
            atlas.isSaved(slide.id) ? '已收藏' : '收藏'
          }}
        </button>
      </div>
    </section>
  </dialog>
</template>
