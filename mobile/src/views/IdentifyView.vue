<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ImagePlus, ScanLine, X } from '@lucide/vue'
import RecognitionResult from '../components/RecognitionResult.vue'
import { fetchCapabilities } from '../services/api'
import { submitRecognition } from '../services/recognition'
import { samples, sampleById } from '../lib/samples'
const router = useRouter()
const route = useRoute()
const available = ref(false)
const serviceText = ref('正在检查识别服务…')
const file = ref<File | null>(null)
const preview = ref('')
const message = ref('')
const busy = ref(false)
const choosing = ref(false)
const isSample = ref(false)
const sampleRoi = ref<number[] | undefined>()
let selection = 0
let active = true
const taskId = computed(() => (typeof route.query.task === 'string' ? route.query.task : ''))
async function connect() {
  serviceText.value = '正在检查识别服务…'
  available.value = false
  try {
    const cap = await fetchCapabilities()
    if (!active) return
    available.value = cap?.classifier.available ?? false
    serviceText.value = available.value ? '单图分类服务已连接' : '识别服务未连接，仍可浏览影像'
  } catch {
    if (active) serviceText.value = '暂时无法连接识别服务'
  }
}
onMounted(connect)
function clear() {
  selection++
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  file.value = null
  message.value = ''
  isSample.value = false
  choosing.value = false
  sampleRoi.value = undefined
}
function choose(selected: File) {
  if (!['image/png', 'image/jpeg'].includes(selected.type)) {
    message.value = '请选择 PNG 或 JPEG 图片。'
    return
  }
  if (selected.size > 8 * 1024 * 1024) {
    message.value = '请选择不超过 8 MB 的图片。'
    return
  }
  file.value = selected
  preview.value = URL.createObjectURL(selected)
}
async function pick(event: Event) {
  const input = event.target as HTMLInputElement
  const selected = input.files?.[0]
  input.value = ''
  if (!selected) return
  clear()
  choose(selected)
  await router.replace('/recognize')
}
async function useSample(id: string) {
  if (busy.value) return
  clear()
  const sample = sampleById(id)
  if (!sample) {
    message.value = '未找到这张影像。'
    return
  }
  const revision = selection
  choosing.value = true
  try {
    const response = await fetch(sample.imageUrl, { signal: AbortSignal.timeout(10000) })
    if (!response.ok) throw new Error('影像加载失败。')
    const blob = await response.blob()
    if (!active || selection !== revision) return
    choose(new File([blob], `${id}.png`, { type: 'image/png' }))
    isSample.value = true
    sampleRoi.value = sample.roi
  } catch (e) {
    if (active && revision === selection)
      message.value = e instanceof Error ? e.message : '图片加载失败。'
  } finally {
    if (active && revision === selection) choosing.value = false
  }
}
watch(
  () => route.query.sample,
  (value) => {
    if (typeof value === 'string') void useSample(value)
  },
  { immediate: true },
)
async function submit() {
  if (!file.value || !available.value || busy.value) return
  busy.value = true
  message.value = ''
  try {
    const task = await submitRecognition(file.value, sampleRoi.value)
    if (active) {
      clear()
      await router.replace({ path: '/recognize', query: { task: task.taskId } })
    }
  } catch (e) {
    if (active)
      message.value = `${e instanceof Error ? e.message : '提交失败。'} 若网络中断，请先到“我的”查看是否已有记录，避免重复提交。`
  } finally {
    if (active) busy.value = false
  }
}
onBeforeUnmount(() => {
  active = false
  clear()
})
</script>
<template>
  <section class="content-view identify-view">
    <div class="page-heading">
      <h1>识别滑坡类型</h1>
      <p>上传单个滑坡的裁片</p>
    </div>
    <div class="service-status">
      <span :class="{ ready: available }"></span>{{ serviceText
      }}<button v-if="!available" class="text-button" @click="connect">重试</button>
    </div>
    <template v-if="taskId">
      <RecognitionResult :key="taskId" :task-id="taskId" />
      <div class="sample-actions">
        <RouterLink to="/recognize" class="primary-button">识别另一张</RouterLink
        ><RouterLink to="/mine" class="secondary-button">查看历史记录</RouterLink>
      </div>
    </template>
    <template v-else>
      <label class="upload-area" :class="{ 'has-image': preview }">
        <img v-if="preview" :src="preview" alt="待识别图片预览" /><template v-else
          ><ImagePlus :size="34" :stroke-width="1.4" /><strong>{{
            choosing ? '正在加载影像…' : '选择滑坡影像'
          }}</strong
          ><span>PNG / JPEG · 最大 8 MB</span></template
        >
        <input
          type="file"
          accept="image/png,image/jpeg"
          aria-label="选择待识别图片"
          :disabled="busy"
          @change="pick"
        />
      </label>
      <div v-if="file" class="file-row">
        <span>{{ file.name }}</span
        ><button class="icon-button" aria-label="移除图片" :disabled="busy" @click="clear">
          <X :size="18" />
        </button>
      </div>
      <p v-if="isSample" class="sample-caption">编目影像 · 使用已有目标框</p>
      <button
        class="primary-button full-width"
        :disabled="!available || !file || busy || choosing"
        @click="submit"
      >
        <ScanLine :size="18" />{{ busy ? '正在提交…' : '开始分类' }}
      </button>
      <p v-if="message" role="alert" class="notice">{{ message }}</p>
      <div class="section-heading sample-picker-heading">
        <h2>从目录选择</h2>
        <RouterLink to="/atlas#samples">全部影像 →</RouterLink>
      </div>
      <div class="sample-picker">
        <button
          v-for="sample in samples.filter((_, i) => i % 2 === 0)"
          :key="sample.id"
          :disabled="busy"
          @click="useSample(sample.id)"
        >
          <img :src="sample.imageUrl" :alt="`选择滑坡 ${sample.sourceId}影像`" /><span
            >{{ sample.type }} · {{ sample.sourceId }}</span
          >
        </button>
      </div>
      <p class="result-note">开始分类后上传图片，结果需复核。</p>
      <details class="quiet-details">
        <summary>使用说明</summary>
        <p>
          只对单个滑坡分类，不检测整幅遥感图。目录图片来自训练数据，试用结果不作为独立测试成绩。
        </p>
      </details>
    </template>
  </section>
</template>
