<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ImagePlus, ScanLine, X } from '@lucide/vue'
import { fetchCapabilities } from '../services/api'
import { submitDetection } from '../services/recognition'
const route = useRoute()
const router = useRouter()
const available = ref(false)
const serviceText = ref('正在检查检测服务…')
const file = ref<File | null>(null)
const preview = ref('')
const dimensions = ref<{ width: number; height: number } | null>(null)
const imageValid = ref(true)
const message = ref('')
const busy = ref(false)
const MAX_SIDE = 2048
const MAX_PIXELS = 2048 * 2048
let active = true
async function connect() {
  serviceText.value = '正在检查检测服务…'
  available.value = false
  try {
    const cap = await fetchCapabilities()
    if (!active) return
    available.value = Boolean(cap?.detection.available)
    serviceText.value = available.value
      ? '检测服务已连接，结果需人工复核'
      : (cap?.detection.reason ?? '检测服务未连接')
  } catch {
    if (active) serviceText.value = '暂时无法连接检测服务'
  }
}
onMounted(connect)
function clear() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  file.value = null
  dimensions.value = null
  imageValid.value = true
  message.value = ''
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
  imageValid.value = true
  preview.value = URL.createObjectURL(selected)
  dimensions.value = null
  const image = new Image()
  image.onload = () => {
    if (file.value !== selected) return
    const width = image.naturalWidth
    const height = image.naturalHeight
    dimensions.value = { width, height }
    if (width * height > MAX_PIXELS || Math.max(width, height) > MAX_SIDE) {
      imageValid.value = false
      message.value = `图片为 ${width}×${height}，超过检测限制：最长边不超过 2048 像素，总像素不超过 2048×2048。`
    }
  }
  image.onerror = () => {
    if (file.value !== selected) return
    imageValid.value = false
    message.value = '无法读取该图片，请重新选择有效的 CTX 裁图。'
  }
  image.src = preview.value
}
function pick(event: Event) {
  const input = event.target as HTMLInputElement
  const selected = input.files?.[0]
  input.value = ''
  if (!selected) return
  clear()
  choose(selected)
}
async function submit() {
  if (!file.value || !imageValid.value || !available.value || busy.value) return
  busy.value = true
  message.value = ''
  try {
    const task = await submitDetection(file.value)
    if (active && route.query.mode === 'detect') {
      clear()
      await router.replace({ path: '/recognize', query: { mode: 'detect', task: task.taskId } })
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
  <div class="recognition-workspace">
    <div class="upload-column">
      <div class="service-status" role="status">
        {{ serviceText }}<button v-if="!available" class="text-button" @click="connect">重试</button>
      </div>
      <label class="upload-area" :class="{ 'has-image': preview }">
        <img v-if="preview" :src="preview" alt="待检测图片预览" /><template v-else
          ><ImagePlus :size="34" :stroke-width="1.4" /><strong>选择 CTX 裁图</strong
          ><span>PNG / JPEG，最大 8 MB；最长边不超过 2048，总像素不超过 2048×2048</span></template
        >
        <input
          type="file"
          accept="image/png,image/jpeg"
          aria-label="选择待检测 CTX 裁图"
          :disabled="busy"
          @change="pick"
        />
      </label>
      <div v-if="file" class="file-row">
        <span class="file-meta"
          >{{ file.name
          }}<em v-if="dimensions">{{ dimensions.width }}×{{ dimensions.height }}</em></span
        ><button class="icon-button" aria-label="移除图片" :disabled="busy" @click="clear">
          <X :size="18" />
        </button>
      </div>
      <button
        class="primary-button full-width"
        :disabled="!available || !file || !imageValid || busy"
        @click="submit"
      >
        <ScanLine :size="18" />{{ busy ? '正在提交…' : '开始检测' }}
      </button>
      <p v-if="message" role="alert" class="notice">{{ message }}</p>
    </div>
    <div class="sample-column">
      <div class="section-heading"><h2>检测流程</h2></div>
      <ol class="detect-steps">
        <li>上传整幅 CTX 裁图，先扫描候选框。</li>
        <li>等待任务完成，约每秒查询一次。</li>
        <li>点选候选框放大，查看逐框四类分数。</li>
      </ol>
      <p class="muted">
        检测需整幅裁片，不能用单目标裁图代替。未检出表示阈值下没有保留框，不代表没有滑坡；结果需人工复核。
      </p>
    </div>
  </div>
</template>
<style scoped>
.detect-steps {
  margin: 0;
  padding-inline-start: 1.25em;
  color: var(--text-secondary);
  line-height: 1.8;
}
</style>
