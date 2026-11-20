<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { sampleById } from '../lib/samples'
import {
  getRecognition,
  recognitionImage,
  taskLabel,
  type RecognitionTask,
} from '../services/recognition'
import { slideTypes, typeInfo } from '../types'
const props = defineProps<{ taskId: string; sampleId?: string }>()
const sample = computed(() => (props.sampleId ? sampleById(props.sampleId) : undefined))
const task = ref<RecognitionTask | null>(null)
const error = ref('')
const image = ref('')
const imageError = ref(false)
const retry = ref(0)
const elapsed = ref(0)
const stopped = ref(false)
const screeningNote = computed(() => {
  const current = task.value
  if (current?.status !== 'completed' || !current.screening) return ''
  return {
    possible_landslide: '',
    no_landslide: '初筛未发现足够的滑坡特征。',
    unsupported_image: '请上传裁好的灰度 CTX 遥感影像。',
    inconclusive: '暂无法确认图中是否有滑坡。',
  }[current.screening.decision]
})
let controller: AbortController | undefined
let ticker: ReturnType<typeof setInterval> | undefined
let runId = 0
function stopTicker() {
  if (ticker) clearInterval(ticker)
  ticker = undefined
}
function stop() {
  controller?.abort()
  stopTicker()
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
    controller = new AbortController()
    const signal = controller.signal
    stopTicker()
    ticker = setInterval(() => {
      elapsed.value += 1
    }, 1000)
    try {
      while (!signal.aborted) {
        const current = await getRecognition(
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
  <section class="recognition-result" aria-label="识别结果">
    <p v-if="!task && !error" role="status">正在读取识别记录…</p>
    <div v-if="error" class="notice" role="alert">
      <p>{{ error }} 查询失败不会取消已提交的任务。</p>
      <button class="text-button" @click="retry++">重新查询</button>
    </div>
    <template v-if="task">
      <img v-if="image" :src="image" alt="本次识别图片" class="result-preview" />
      <p v-if="imageError" class="muted">预览暂时无法读取，结果仍可查看。</p>
      <template v-if="screeningNote">
        <h2 aria-live="polite">{{ taskLabel(task) }}</h2>
        <p>{{ screeningNote }}</p>
      </template>
      <span v-else class="status-pill" role="status">{{ taskLabel(task) }}</span>
      <p
        v-if="!stopped && (task.status === 'queued' || task.status === 'running')"
        class="muted recognition-wait"
      >
        已等待 {{ elapsed }} 秒，约每秒查询一次；可以离开此页，在“我的”中继续查看，无需重复提交。
        <button class="text-button" @click="halt">停止等待</button>
      </p>
      <p v-if="stopped" class="muted" role="status">
        已停止等待。识别在服务端继续进行，稍后到“我的”查看结果。
        <button class="text-button" @click="retry++">继续查询</button>
      </p>
      <p v-if="task.status === 'failed'" role="alert">
        {{
          task.errorCode === 'PROCESS_RESTARTED'
            ? '服务重启中断了这次识别，请重新选择图片提交。'
            : '本次识别没有成功，请重新选择图片提交。'
        }}
      </p>
      <div v-if="sample && task.status === 'completed'" class="sample-reference">
        <p>
          <RouterLink
            :to="{ name: 'catalog-record', params: { id: sample.id }, query: { from: 'atlas' } }"
          >
            图鉴 #{{ sample.sourceId }}
          </RouterLink>
          <span>编目类型 · {{ typeInfo[sample.type].name }}（{{ sample.type }}）</span>
        </p>
        <p v-if="task.screening?.decision === 'no_landslide'">这处滑坡已收录，本次初筛漏检。</p>
      </div>
      <component
        :is="task.screening ? 'div' : 'details'"
        v-if="
          task.status === 'completed' &&
          task.result &&
          (!task.screening || task.screening.decision === 'possible_landslide')
        "
        :class="{ 'result-details': !task.screening }"
      >
        <summary v-if="!task.screening">查看旧版类型分数</summary>
        <h2>
          {{ typeInfo[task.result.prediction].name }} <span>{{ task.result.prediction }}</span>
        </h2>
        <p>这是模型倾向的类别，需结合影像复核。</p>
        <div class="score-list">
          <div v-for="type in slideTypes" :key="type" class="score-row">
            <span
              >{{ typeInfo[type].name }} <small>{{ type }}</small></span
            >
            <meter
              :value="task.result.probabilities[type]"
              min="0"
              max="1"
              :aria-label="`${type} 分类分数`"
            />
            <strong>{{ (task.result.probabilities[type] * 100).toFixed(1) }}%</strong>
          </div>
        </div>
        <p class="result-note">分数表示模型对四种类型的相对倾向。</p>
        <RouterLink :to="`/atlas/type/${task.result.prediction}`" class="secondary-button"
          >查看这一类的特征与区别 →</RouterLink
        >
        <details class="result-details">
          <summary>模型与处理信息</summary>
          <p v-if="task.screening?.modelVersion">初筛模型 {{ task.screening.modelVersion }}</p>
          <p>类型模型 {{ task.modelVersion }}</p>
          <p>预处理：{{ task.preprocessingVersion }}</p>
        </details>
      </component>
      <p class="result-note">
        {{ new Date(task.createdAt).toLocaleString('zh-CN') }} · 已保存到本次会话的识别记录
      </p>
    </template>
  </section>
</template>

<style scoped>
.sample-reference {
  margin-block: var(--space-sm);
  padding-inline-start: var(--space-sm);
  border-inline-start: 1px solid var(--line);
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.sample-reference p:first-child {
  display: flex;
  flex-wrap: wrap;
  gap: 0 var(--space-sm);
}
.sample-reference a {
  color: var(--accent-hover);
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
</style>
