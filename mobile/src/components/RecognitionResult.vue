<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import {
  getRecognition,
  recognitionImage,
  taskLabel,
  type RecognitionTask,
} from '../services/recognition'
import { slideTypes, typeInfo } from '../types'
const props = defineProps<{ taskId: string }>()
const task = ref<RecognitionTask | null>(null)
const error = ref('')
const image = ref('')
const imageError = ref(false)
const retry = ref(0)
let controller: AbortController | undefined
function stop() {
  controller?.abort()
  if (image.value) URL.revokeObjectURL(image.value)
  image.value = ''
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
    stop()
    task.value = null
    error.value = ''
    imageError.value = false
    controller = new AbortController()
    const signal = controller.signal
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
      <span class="status-pill" role="status">{{ taskLabel(task) }}</span>
      <p v-if="task.status === 'queued' || task.status === 'running'" class="muted">
        可以离开此页，在“我的”中继续查看，无需重复提交。
      </p>
      <p v-if="task.status === 'failed'" role="alert">
        {{
          task.errorCode === 'PROCESS_RESTARTED'
            ? '服务重启中断了这次识别，请重新选择图片提交。'
            : '本次识别没有成功，请重新选择图片提交。'
        }}
      </p>
      <template v-if="task.result">
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
        <p class="result-note">这些是分类分数，不代表正确率，也不判断图片中是否存在滑坡。</p>
        <RouterLink :to="`/atlas/type/${task.result.prediction}`" class="secondary-button"
          >查看这一类的特征与区别 →</RouterLink
        >
        <details class="result-details">
          <summary>模型与处理信息</summary>
          <p>模型：{{ task.modelVersion }}</p>
          <p>预处理：{{ task.preprocessingVersion }}</p>
        </details>
      </template>
      <p class="result-note">
        {{ new Date(task.createdAt).toLocaleString('zh-CN') }} · 已保存到本次会话的识别记录
      </p>
    </template>
  </section>
</template>
