<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { ImagePlus, ScanLine, Info, X } from '@lucide/vue'
import PageHeading from '../components/PageHeading.vue'
import { fetchCapabilities, classifyImage } from '../services/api'
const available = ref(false)
const serviceText = ref('识别服务尚未接入')
const file = ref<File | null>(null)
const preview = ref('')
const message = ref('')
const busy = ref(false)
const result = ref('')
onMounted(async () => {
  try {
    const cap = await fetchCapabilities()
    available.value = cap?.classifier.available ?? false
    if (available.value) serviceText.value = '单图分类服务已连接'
  } catch {
    serviceText.value = '暂时无法连接识别服务'
  }
})
function clear() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  file.value = null
  result.value = ''
  message.value = ''
}
function pick(event: Event) {
  const selected = (event.target as HTMLInputElement).files?.[0]
  clear()
  if (!selected) return
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(selected.type)) {
    message.value = '请选择 PNG、JPEG 或 WebP 图片。'
    return
  }
  if (selected.size > 10 * 1024 * 1024) {
    message.value = '请选择不超过10 MB的图片。'
    return
  }
  file.value = selected
  preview.value = URL.createObjectURL(selected)
}
async function submit() {
  if (!file.value || !available.value || busy.value) return
  busy.value = true
  message.value = ''
  try {
    const response = (await classifyImage(file.value)) as {
      prediction?: string
      uncertain?: boolean
    }
    if (!response.prediction || !['SL', 'SP', 'FL', 'CL'].includes(response.prediction))
      throw new Error('服务返回的数据无法识别')
    result.value = response.uncertain ? '需要人工复核' : response.prediction
  } catch (e) {
    message.value = e instanceof Error ? e.message : '识别失败'
  } finally {
    busy.value = false
  }
}
onBeforeUnmount(clear)
</script>
<template>
  <section class="content-view">
    <PageHeading
      eyebrow="A CLOSER LOOK"
      title="滑坡类型识别"
      description="上传已裁剪的单个滑坡图片，查看类型识别结果。"
    />
    <div class="service-status"><span :class="{ ready: available }"></span>{{ serviceText }}</div>
    <label class="upload-area" :class="{ 'has-image': preview }"
      ><img v-if="preview" :src="preview" alt="待识别图片预览" /><template v-else
        ><ImagePlus :size="37" :stroke-width="1" /><strong>选择一张影像</strong
        ><span>PNG / JPEG / WebP · 最大10 MB</span></template
      ><input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        aria-label="选择待识别图片"
        @change="pick"
    /></label>
    <div v-if="file" class="file-row">
      <span>{{ file.name }}</span
      ><button class="icon-button" aria-label="移除图片" @click="clear"><X :size="18" /></button>
    </div>
    <button
      class="primary-button full-width"
      :disabled="!available || !file || busy"
      @click="submit"
    >
      <ScanLine :size="18" />{{
        busy ? '识别中…' : available ? '识别滑坡类型' : '等待识别服务接入'
      }}
    </button>
    <p v-if="message" role="alert" class="notice">{{ message }}</p>
    <div v-if="result" class="result-card">
      <span class="eyebrow">识别结果</span>
      <h2>{{ result }}</h2>
      <p>模型结果仅作为辅助判断。</p>
    </div>
    <div class="info-card">
      <Info :size="18" />
      <p>
        此入口用于单个滑坡的类型分类。大幅影像中的滑坡检测与资料问答将在后续接入。服务未连接时，所选图片只在本机预览。
      </p>
    </div>
  </section>
</template>