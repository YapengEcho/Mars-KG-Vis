<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ImagePlus, ScanLine, Info, X } from '@lucide/vue'
import PageHeading from '../components/PageHeading.vue'
import { classifyImage, fetchCapabilities, RecognitionError } from '../services/api'
import { formatPercent, type ClassificationResult } from '../lib/recognition'
import { recognitions } from '../stores/recognitions'
import { slideTypes, typeInfo } from '../types'

const maxBytes = 8 * 1024 * 1024
const maxPixels = 4096
const serviceState = ref<'checking' | 'ready' | 'unavailable' | 'offline'>('checking')
const serviceText = computed(() => {
  if (serviceState.value === 'checking') return '正在检查识别服务…'
  if (serviceState.value === 'ready') return '单图分类服务已连接'
  if (serviceState.value === 'offline') return '尚未连接后端服务，当前只能预览图片'
  return '识别服务尚未接入，当前只能预览图片'
})
const available = computed(() => serviceState.value === 'ready')
const file = ref<File | null>(null)
const preview = ref('')
const dimensions = ref<{ width: number; height: number } | null>(null)
const imageValid = ref(false)
const message = ref('')
const busy = ref(false)
const elapsed = ref(0)
const result = ref<ClassificationResult | null>(null)
const saved = ref(false)
let timer: ReturnType<typeof setInterval> | undefined
let controller: AbortController | undefined

async function checkService() {
  serviceState.value = 'checking'
  try {
    const cap = await fetchCapabilities()
    if (!cap) serviceState.value = 'offline'
    else serviceState.value = cap.classifier.available ? 'ready' : 'unavailable'
  } catch {
    serviceState.value = 'offline'
  }
}
onMounted(checkService)

function releasePreview() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
}
function clear() {
  if (busy.value) return
  releasePreview()
  file.value = null
  dimensions.value = null
  imageValid.value = false
  result.value = null
  saved.value = false
  message.value = ''
}
function pick(event: Event) {
  const input = event.target as HTMLInputElement
  const selected = input.files?.[0]
  clear()
  input.value = ''
  if (!selected) return
  if (!['image/png', 'image/jpeg'].includes(selected.type)) {
    message.value = '请选择单帧 PNG 或 JPEG 图片。'
    return
  }
  if (selected.size > maxBytes) {
    message.value = '图片超过 8 MB，请先裁剪后再上传。'
    return
  }
  file.value = selected
  preview.value = URL.createObjectURL(selected)
  const image = new Image()
  image.onload = () => {
    dimensions.value = { width: image.naturalWidth, height: image.naturalHeight }
    imageValid.value = image.naturalWidth <= maxPixels && image.naturalHeight <= maxPixels
    if (!imageValid.value)
      message.value = `图片为 ${image.naturalWidth}×${image.naturalHeight}，超过 ${maxPixels}×${maxPixels} 上限，请先裁剪出单个滑坡。`
  }
  image.onerror = () => {
    imageValid.value = false
    message.value = '无法读取该图片，请重新选择有效的裁图。'
  }
  image.src = preview.value
}

function startTimer() {
  elapsed.value = 0
  timer = setInterval(() => {
    elapsed.value += 1
  }, 1000)
}
function stopTimer() {
  if (timer) clearInterval(timer)
  timer = undefined
  controller = undefined
}
function cancelSubmit() {
  controller?.abort()
}
async function makeThumbnail(source: File): Promise<string> {
  try {
    const bitmap = await createImageBitmap(source)
    const scale = Math.min(1, 360 / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale))
    canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const context = canvas.getContext('2d')
    if (!context) return ''
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    bitmap.close()
    return canvas.toDataURL('image/jpeg', 0.72)
  } catch {
    return ''
  }
}
async function submit() {
  const selected = file.value
  if (!selected || !imageValid.value || !available.value || busy.value) return
  busy.value = true
  message.value = ''
  result.value = null
  saved.value = false
  startTimer()
  const current = new AbortController()
  controller = current
  try {
    const classified = await classifyImage(selected, current.signal)
    result.value = classified
    const image = await makeThumbnail(selected)
    recognitions.add(classified, selected.name, image)
    saved.value = true
  } catch (error) {
    if (current.signal.aborted) {
      message.value = '已取消本次识别，图片仍保留在本机。'
    } else if (error instanceof RecognitionError) {
      message.value = error.message
    } else {
      message.value = '识别失败，请稍后重试。'
    }
  } finally {
    busy.value = false
    stopTimer()
  }
}
onBeforeUnmount(() => {
  controller?.abort()
  stopTimer()
  releasePreview()
})
</script>

<template>
  <section class="content-view">
    <PageHeading
      eyebrow="A CLOSER LOOK"
      title="滑坡类型识别"
      description="上传单张滑坡裁图，得到四分类结果，并保存到本机回看。"
    />
    <div class="service-status">
      <span :class="{ ready: available }"></span>{{ serviceText }}
      <button
        v-if="serviceState === 'offline' || serviceState === 'unavailable'"
        class="text-button"
        @click="checkService"
      >
        重试连接
      </button>
    </div>
    <label class="upload-area" :class="{ 'has-image': preview }"
      ><img v-if="preview" :src="preview" alt="待识别图片预览" /><template v-else
        ><ImagePlus :size="37" :stroke-width="1" /><strong>选择一张影像</strong
        ><span>单帧 PNG / JPEG · 最大 8 MB · 不超过 4096×4096</span></template
      ><input
        type="file"
        accept="image/png,image/jpeg"
        aria-label="选择待识别图片"
        :disabled="busy"
        @change="pick"
    /></label>
    <div v-if="file" class="file-row">
      <span class="file-meta"
        >{{ file.name
        }}<em v-if="dimensions">{{ dimensions.width }}×{{ dimensions.height }}</em></span
      ><button class="icon-button" aria-label="移除图片" :disabled="busy" @click="clear">
        <X :size="18" />
      </button>
    </div>
    <button v-if="busy" class="primary-button full-width" disabled>
      <ScanLine :size="18" />识别中… {{ elapsed }}s
    </button>
    <button
      v-else
      class="primary-button full-width"
      :disabled="!available || !file || !imageValid"
      @click="submit"
    >
      <ScanLine :size="18" />{{
        available ? (file ? '识别滑坡类型' : '请先选择图片') : '等待识别服务接入'
      }}
    </button>
    <p v-if="busy" class="recognition-wait">
      已提交，等待服务返回结果，最长 65 秒。
      <button class="text-button" @click="cancelSubmit">取消等待</button>
    </p>
    <p v-if="message" role="alert" class="notice">
      {{ message }}
      <button v-if="file && imageValid && available && !busy" class="text-button" @click="submit">
        重新识别
      </button>
    </p>
    <p v-if="recognitions.state.storageError" class="storage-warning" role="status">
      {{ recognitions.state.storageError }}
    </p>
    <div v-if="result" class="result-card recognition-result">
      <div class="result-head">
        <div>
          <span class="eyebrow">识别结果</span>
          <h2>
            <i class="type-swatch" :style="{ background: typeInfo[result.prediction].color }"></i
            >{{ typeInfo[result.prediction].name }} · {{ result.prediction }}
          </h2>
          <p>置信度 {{ formatPercent(result.confidence) }} · 模型版本 {{ result.modelVersion }}</p>
        </div>
        <span class="result-badge" :class="{ uncertain: result.uncertain }">{{
          result.uncertain ? '待复核' : '已分类'
        }}</span>
      </div>
      <ul class="probability-list">
        <li v-for="type in slideTypes" :key="type">
          <span class="probability-name">{{ type }} · {{ typeInfo[type].name }}</span>
          <span class="probability-value">{{ formatPercent(result.probabilities[type]) }}</span>
          <span class="probability-track"
            ><i
              :style="{
                width: formatPercent(result.probabilities[type]),
                background: typeInfo[type].color,
              }"
            ></i
          ></span>
        </li>
      </ul>
      <p class="result-note">
        {{
          result.uncertain
            ? '模型置信度较低，已标记为待复核，请结合影像与资料判断。'
            : '概率为模型输出分数，未经校准，不作为地质真实性的百分比。'
        }}
      </p>
      <p v-if="saved" class="result-saved">已保存到本机「我的 · 识别记录」，可随时回看。</p>
      <div class="result-actions">
        <RouterLink :to="`/atlas/type/${result.prediction}`" class="secondary-button"
          >查看{{ typeInfo[result.prediction].name }}图鉴</RouterLink
        ><RouterLink to="/mine" class="secondary-button">识别记录</RouterLink>
      </div>
    </div>
    <div class="info-card">
      <Info :size="18" />
      <p>
        此入口只处理单个滑坡裁图，结果仅作辅助判断；大幅影像的滑坡检测与资料问答留待后续接入。服务未连接时，所选图片只在本机预览，不会上传。
      </p>
    </div>
  </section>
</template>
