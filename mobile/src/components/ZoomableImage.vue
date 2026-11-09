<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { Maximize2, Minus, Plus } from '@lucide/vue'

const props = withDefaults(defineProps<{ src: string; alt: string; maxScale?: number }>(), {
  maxScale: 4,
})

const scale = ref(1)
const offset = ref({ x: 0, y: 0 })
const dragging = ref(false)
const pointers = new Map<number, { x: number; y: number }>()
let pinchStart: { distance: number; scale: number } | null = null

function clampScale(value: number): number {
  return Math.min(props.maxScale, Math.max(1, value))
}

function zoom(next: number): void {
  scale.value = clampScale(next)
  if (scale.value === 1) offset.value = { x: 0, y: 0 }
}

function reset(): void {
  scale.value = 1
  offset.value = { x: 0, y: 0 }
}

function toggle(): void {
  zoom(scale.value > 1 ? 1 : 2)
}

function onWheel(event: WheelEvent): void {
  zoom(scale.value * (event.deltaY < 0 ? 1.2 : 1 / 1.2))
}

function pointerDistance(): number {
  const [a, b] = [...pointers.values()]
  if (!a || !b) return 0
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function onPointerDown(event: PointerEvent): void {
  ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  dragging.value = true
  if (pointers.size === 2) pinchStart = { distance: pointerDistance(), scale: scale.value }
}

function onPointerMove(event: PointerEvent): void {
  const previous = pointers.get(event.pointerId)
  if (!previous) return
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  if (pointers.size === 2 && pinchStart) {
    zoom(pinchStart.scale * (pointerDistance() / (pinchStart.distance || 1)))
    return
  }
  if (scale.value > 1) {
    offset.value = {
      x: offset.value.x + (event.clientX - previous.x),
      y: offset.value.y + (event.clientY - previous.y),
    }
  }
}

function onPointerUp(event: PointerEvent): void {
  pointers.delete(event.pointerId)
  if (pointers.size < 2) pinchStart = null
  if (pointers.size === 0) dragging.value = false
}

watch(
  () => props.src,
  () => reset(),
)

onBeforeUnmount(() => pointers.clear())
</script>

<template>
  <div
    class="zoom-viewer"
    :class="{ 'is-zoomed': scale > 1, 'is-dragging': dragging }"
    @wheel.prevent="onWheel"
    @dblclick="toggle"
  >
    <img
      :src="src"
      :alt="alt"
      :style="{
        transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${scale})`,
      }"
      draggable="false"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
    />
    <div class="zoom-controls">
      <button
        type="button"
        class="icon-button"
        aria-label="缩小影像"
        :disabled="scale <= 1"
        @click.stop="zoom(scale / 1.4)"
      >
        <Minus :size="16" />
      </button>
      <span class="zoom-level" aria-live="polite">{{ Math.round(scale * 100) }}%</span>
      <button
        type="button"
        class="icon-button"
        aria-label="放大影像"
        :disabled="scale >= maxScale"
        @click.stop="zoom(scale * 1.4)"
      >
        <Plus :size="16" />
      </button>
      <button
        type="button"
        class="icon-button"
        aria-label="复位影像"
        :disabled="scale === 1 && offset.x === 0 && offset.y === 0"
        @click.stop="reset"
      >
        <Maximize2 :size="16" />
      </button>
    </div>
    <p class="zoom-hint">滚轮或双指缩放 · 拖动查看细节 · 双击放大</p>
  </div>
</template>
