<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { Maximize2 } from '@lucide/vue'
import { fitCandidateBox } from '../lib/detection'
import {
  getDetection,
  recognitionImage,
  taskLabel,
  type DetectionCandidate,
  type DetectionTask,
} from '../services/recognition'
import { slideTypes, typeInfo } from '../types'
const props = defineProps<{ taskId: string }>()
const task = ref<DetectionTask | null>(null)
const error = ref('')
const image = ref('')
const imageError = ref(false)
const retry = ref(0)
const elapsed = ref(0)
const stopped = ref(false)
const selectedId = ref('')
const stage = ref<HTMLElement>()
const stageWidth = ref(0)
const candidates = computed(() => task.value?.result?.candidates ?? [])
const selected = computed<DetectionCandidate | undefined>(() =>
  candidates.value.find((candidate) => candidate.candidateId === selectedId.value),
)
const input = computed(() => task.value?.input)
const transform = computed(() => {
  if (!selected.value || !input.value || !stageWidth.value) return { scale: 1, x: 0, y: 0 }
  return fitCandidateBox(selected.value.box, input.value, stageWidth.value)
})
const canvasStyle = computed(() => ({
  transform: `translate(${transform.value.x}px, ${transform.value.y}px) scale(${transform.value.scale})`,
}))
const outcomeNote = computed(() => {
  if (task.value?.status !== 'completed' || !task.value.result) return ''
  if (task.value.result.outcome === 'no_detections')
    return '本次没有检出的候选。阈值下未保留框，不代表影像中没有滑坡。'
  return task.value.result.truncated
    ? '候选数量超过上限，仅显示并分类了前 12 个框。'
    : '检测框和类型均需人工复核，分数不能单独确认地质类别。'
})
let controller: AbortController | undefined
let ticker: ReturnType<typeof setInterval> | undefined
let observer: ResizeObserver | undefined
let runId = 0
function stopTicker() {
  if (ticker) clearInterval(ticker)
  ticker = undefined
}
function stop() {
  controller?.abort()
  stopTicker()
  observer?.disconnect()
  observer = undefined
  if (image.value) URL.revokeObjectURL(image.value)
  image.value = ''
}
function halt() {
  stopped.value = true
  stopTicker()
  controller?.abort()
}
function delay(signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    const finish = () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', finish)
      resolve()
    }
    const timer = setTimeout(finish, 1000)
    signal.addEventListener('abort', finish, { once: true })
  })
}
function select(candidateId: string) {
  selectedId.value = selectedId.value === candidateId ? '' : candidateId
}
watch(
  stage,
  (element) => {
    observer?.disconnect()
    observer = undefined
    if (!element) return
    stageWidth.value = element.clientWidth
    observer = new ResizeObserver(([entry]) => {
      stageWidth.value = entry.contentRect.width
    })
    observer.observe(element)
  },
  { flush: 'post' },
)
watch(
  () => [props.taskId, retry.value],
  async () => {
    const run = ++runId
    stop()
    task.value = null
    error.value = ''
    imageError.value = false
    stopped.value = false
    elapsed.value = 0
    selectedId.value = ''
    controller = new AbortController()
    const signal = controller.signal
    ticker = setInterval(() => {
      elapsed.value += 1
    }, 1000)
    try {
      while (!signal.aborted) {
        const current = await getDetection(
          props.taskId,
          AbortSignal.any([signal, AbortSignal.timeout(15000)]),
        )
        if (signal.aborted) return
        task.value = current
        if (!image.value && !imageError.value) {
          try {
            const blob = await recognitionImage(
              props.taskId,
              AbortSignal.any([signal, AbortSignal.timeout(15000)]),
            )
            if (signal.aborted) return
            image.value = URL.createObjectURL(blob)
          } catch {
            if (!signal.aborted) imageError.value = true
          }
        }
        if (current.status === 'completed' || current.status === 'failed') break
        await delay(signal)
      }
    } catch (e) {
      if (!signal.aborted) error.value = e instanceof Error ? e.message : '暂时无法读取结果。'
    } finally {
      if (run === runId) stopTicker()
    }
  },
  { immediate: true },
)
onBeforeUnmount(stop)
</script>
<template>
  <section class="detection-result" aria-label="检测结果">
    <p v-if="!task && !error" role="status">正在读取检测记录…</p>
    <div v-if="error" class="notice" role="alert">
      <p>{{ error }} 查询失败不会取消已提交的任务。</p>
      <button class="text-button" @click="retry++">重新查询</button>
    </div>
    <template v-if="task">
      <span class="status-pill" role="status">{{ taskLabel(task) }}</span>
      <p
        v-if="!stopped && (task.status === 'queued' || task.status === 'running')"
        class="muted recognition-wait"
      >
        已等待 {{ elapsed }} 秒，约每秒查询一次；可以离开此页，在“我的”中继续查看，无需重复提交。
        <button class="text-button" @click="halt">停止等待</button>
      </p>
      <p v-if="stopped" class="muted" role="status">
        已停止等待。检测在服务端继续进行，稍后到“我的”查看结果。
        <button class="text-button" @click="retry++">继续查询</button>
      </p>
      <p v-if="task.status === 'failed'" role="alert">
        {{
          task.errorCode === 'PROCESS_RESTARTED'
            ? '服务重启中断了这次检测，请重新选择图片提交。'
            : '本次检测没有成功，请重新选择图片提交。'
        }}
      </p>
      <div v-if="image && task.status === 'completed' && input" ref="stage" class="detection-stage">
        <div class="detection-canvas" :style="canvasStyle">
          <img :src="image" alt="本次检测的 CTX 裁图" />
          <svg
            class="detection-boxes"
            :viewBox="`0 0 ${input.width} ${input.height}`"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            <rect
              v-for="(candidate, index) in candidates"
              :key="candidate.candidateId"
              :x="candidate.box[0]"
              :y="candidate.box[1]"
              :width="candidate.box[2] - candidate.box[0]"
              :height="candidate.box[3] - candidate.box[1]"
              :class="{ selected: candidate.candidateId === selectedId }"
              @click="select(candidate.candidateId)"
            >
              <title>候选 {{ index + 1 }}</title>
            </rect>
          </svg>
        </div>
      </div>
      <img v-else-if="image" :src="image" alt="本次检测的 CTX 裁图" class="result-preview" />
      <p v-if="imageError" class="muted">预览暂时无法读取，结果仍可查看。</p>
      <p v-if="outcomeNote" class="result-note">{{ outcomeNote }}</p>
      <template v-if="task.status === 'completed' && candidates.length">
        <div class="candidate-list" role="list">
          <button
            v-for="(candidate, index) in candidates"
            :key="candidate.candidateId"
            type="button"
            role="listitem"
            :class="{ selected: candidate.candidateId === selectedId }"
            @click="select(candidate.candidateId)"
          >
            <span class="candidate-index">候选 {{ index + 1 }}</span>
            <strong>{{ typeInfo[candidate.classification.prediction].name }}</strong>
            <span class="candidate-score"
              >检测 {{ (candidate.detectionScore * 100).toFixed(0) }}%</span
            >
          </button>
        </div>
        <button v-if="selected" class="text-button" type="button" @click="selectedId = ''">
          <Maximize2 :size="14" />查看全图
        </button>
        <div v-if="selected" class="candidate-detail">
          <h2>
            {{ typeInfo[selected.classification.prediction].name }}
            <span>{{ selected.classification.prediction }}</span>
          </h2>
          <p class="result-note">
            像素框 [{{ selected.box.join(', ') }}] · 检测分数
            {{ (selected.detectionScore * 100).toFixed(1) }}% · 分类置信度
            {{ (selected.classification.confidence * 100).toFixed(1) }}%
          </p>
          <div class="score-list">
            <div v-for="type in slideTypes" :key="type" class="score-row">
              <span>{{ typeInfo[type].name }} <small>{{ type }}</small></span>
              <meter
                :value="selected.classification.probabilities[type]"
                min="0"
                max="1"
                :aria-label="`${type} 分类分数`"
              />
              <strong>{{ (selected.classification.probabilities[type] * 100).toFixed(1) }}%</strong>
            </div>
          </div>
          <p class="result-note">检测分数与类型分数分开计算，均表示模型倾向，需结合影像复核。</p>
          <RouterLink
            :to="`/atlas/type/${selected.classification.prediction}`"
            class="secondary-button"
            >查看这一类的特征与区别 →</RouterLink
          >
        </div>
        <p v-else class="muted">点选影像中的框或上方列表，可放大查看该候选并查看四类分数。</p>
      </template>
      <details v-if="task.status === 'completed'" class="result-details">
        <summary>模型与处理信息</summary>
        <p>检测模型 {{ task.modelVersion }}</p>
        <p>类型模型 {{ candidates[0]?.classification.modelVersion ?? '—' }}</p>
        <p>预处理：{{ task.preprocessingVersion }}</p>
        <p v-if="input">输入尺寸 {{ input.width }}×{{ input.height }} 像素</p>
      </details>
      <p class="result-note">
        {{ new Date(task.createdAt).toLocaleString('zh-CN') }} · 已保存到本次会话的检测记录
      </p>
    </template>
  </section>
</template>
<style scoped>
.detection-stage {
  position: relative;
  margin-block: var(--space-sm);
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #000;
}
.detection-canvas {
  position: relative;
  transform-origin: 0 0;
  transition: transform 0.25s ease;
  will-change: transform;
}
.detection-canvas img {
  display: block;
  width: 100%;
  height: auto;
}
.detection-boxes {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.detection-boxes rect {
  fill: rgba(255, 178, 122, 0.12);
  stroke: #ffb27a;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  cursor: pointer;
}
.detection-boxes rect.selected {
  fill: rgba(124, 196, 255, 0.2);
  stroke: #7cc4ff;
  stroke-width: 2.5;
}
.candidate-list {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs);
  margin-block: var(--space-sm);
}
.candidate-list button {
  display: flex;
  gap: 0.5em;
  align-items: baseline;
  padding: 0.45em 0.7em;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
}
.candidate-list button.selected {
  border-color: var(--accent-hover);
  background: color-mix(in srgb, var(--accent-hover) 14%, transparent);
}
.candidate-index,
.candidate-score {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.candidate-detail h2 span {
  color: var(--text-secondary);
  font-size: 0.7em;
}
</style>
