<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue'
import { ChevronLeft, ChevronRight } from '@lucide/vue'
const props = defineProps<{ count: number; label: string }>()
const id = useId()
const track = ref<HTMLElement>()
const current = ref(0)
const progress = ref(0)
const thumb = ref(100)
const canBack = ref(false)
const canNext = ref(false)
let observer: ResizeObserver | undefined
function update() {
  const el = track.value
  if (!el) return
  const max = Math.max(0, el.scrollWidth - el.clientWidth)
  canBack.value = el.scrollLeft > 2
  canNext.value = el.scrollLeft < max - 2
  thumb.value = el.scrollWidth ? Math.min(100, (el.clientWidth / el.scrollWidth) * 100) : 100
  progress.value = max ? (el.scrollLeft / max) * (100 - thumb.value) : 0
  const children = [...el.children] as HTMLElement[]
  current.value =
    max && !canNext.value
      ? children.length - 1
      : children.reduce(
          (best, item, index) =>
            Math.abs(item.offsetLeft - el.scrollLeft) <
            Math.abs((children[best]?.offsetLeft ?? 0) - el.scrollLeft)
              ? index
              : best,
          0,
        )
}
function move(direction: number) {
  const el = track.value
  if (!el) return
  const target = el.children[Math.max(0, Math.min(props.count - 1, current.value + direction))] as
    HTMLElement | undefined
  if (target)
    el.scrollTo({
      left: target.offsetLeft,
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
    })
}
function keyboard(event: KeyboardEvent) {
  if (event.target !== track.value) return
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    event.preventDefault()
    move(
      event.key === 'Home'
        ? -props.count
        : event.key === 'End'
          ? props.count
          : event.key === 'ArrowRight'
            ? 1
            : -1,
    )
  }
}
watch(
  () => props.count,
  () => void nextTick(update),
)
onMounted(() => {
  observer = new ResizeObserver(update)
  if (track.value) observer.observe(track.value)
  update()
})
onBeforeUnmount(() => observer?.disconnect())
</script>
<template>
  <div class="card-rail">
    <div
      :id="id"
      ref="track"
      class="card-rail-track"
      role="region"
      :aria-label="label"
      tabindex="0"
      @scroll.passive="update"
      @keydown="keyboard"
    >
      <slot />
    </div>
    <div v-if="count > 1" class="rail-navigation">
      <div class="rail-progress" aria-hidden="true">
        <i :style="{ width: `${thumb}%`, left: `${progress}%` }"></i>
      </div>
      <span class="rail-count" aria-live="polite">{{ current + 1 }} / {{ count }}</span>
      <button
        class="icon-button"
        :aria-controls="id"
        aria-label="上一张滑坡"
        :disabled="!canBack"
        @click="move(-1)"
      >
        <ChevronLeft :size="18" />
      </button>
      <button
        class="icon-button"
        :aria-controls="id"
        aria-label="下一张滑坡"
        :disabled="!canNext"
        @click="move(1)"
      >
        <ChevronRight :size="18" />
      </button>
    </div>
  </div>
</template>
