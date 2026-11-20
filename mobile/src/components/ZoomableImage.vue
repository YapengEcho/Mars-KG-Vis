<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { Maximize2, Minus, Plus } from '@lucide/vue'

const props = withDefaults(
  defineProps<{ src: string; alt: string; maxScale?: number; square?: boolean }>(),
  { maxScale: 4 },
)

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
  if (scale.value === 1 && !event.ctrlKey && !event.metaKey) return
  event.preventDefault()
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
    :class="{ 'is-zoomed': scale > 1, 'is-dragging': dragging, 'is-square': square }"
  >
    <div class="zoom-viewport" @wheel="onWheel" @dblclick="toggle">
      <img
        :src="src"
        :alt="alt"
        referrerpolicy="no-referrer"
        :style="{
          transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${scale})`,
        }"
        draggable="false"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
        @pointercancel="onPointerUp"
      />
    </div>
    <div class="zoom-controls" role="group" aria-label="缩放控制">
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
  </div>
</template>

<style scoped>
.zoom-viewer {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  height: 100%;
  min-width: 0;
  overflow: hidden;
  border-radius: 7px;
  border: 1px solid var(--line);
  background: var(--color-paper-3);
}
.zoom-viewer.is-square {
  grid-template-rows: auto auto;
  height: auto;
}
.zoom-viewport {
  position: relative;
  width: 100%;
  min-height: 0;
  overflow: hidden;
  cursor: zoom-in;
  touch-action: pan-y;
}
.is-square .zoom-viewport {
  aspect-ratio: 1;
}
.is-zoomed .zoom-viewport {
  cursor: grab;
  touch-action: none;
}
.is-dragging.is-zoomed .zoom-viewport {
  cursor: grabbing;
}
.zoom-viewport img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
}
.zoom-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-top: 1px solid var(--line);
  background: var(--surface);
}
.zoom-controls .icon-button {
  width: 44px;
  height: 44px;
}
.zoom-controls .icon-button:disabled {
  opacity: 0.4;
}
.zoom-level {
  min-width: 45px;
  color: var(--text-secondary);
  font-size: 0.8rem;
  text-align: center;
  font-variant-numeric: tabular-nums;
}
</style>
