<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Bookmark, ArrowRight, Info } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import PageHeading from '../components/PageHeading.vue'
import { atlas } from '../stores/atlas'
import { apiBase, fetchCapabilities } from '../services/api'
import {
  recognitionHistory,
  deleteRecognition,
  taskLabel,
  resetRecognitionSession,
  RecognitionError,
  type RecognitionTask,
} from '../services/recognition'
const tasks = ref<RecognitionTask[]>([])
const total = ref(0)
const page = ref(1)
const loading = ref(false)
const error = ref('')
const expired = ref(false)
const deleting = ref('')
const connected = ref(false)
async function load(more = false) {
  if (loading.value) return
  loading.value = true
  error.value = ''
  expired.value = false
  try {
    const next = more ? page.value + 1 : 1
    const result = await recognitionHistory(next)
    tasks.value = more ? [...tasks.value, ...result.items] : result.items
    total.value = result.total
    page.value = next
  } catch (e) {
    error.value = e instanceof Error ? e.message : '读取记录失败。'
    expired.value = e instanceof RecognitionError && e.code === 'AUTH_REQUIRED'
  } finally {
    loading.value = false
  }
}
async function remove(id: string) {
  if (deleting.value || !window.confirm('删除这条识别记录和预览图片？此操作无法恢复。')) return
  deleting.value = id
  try {
    await deleteRecognition(id)
    await load()
  } catch (e) {
    error.value = e instanceof Error ? e.message : '删除失败。'
  } finally {
    deleting.value = ''
  }
}
function resetSession() {
  if (!window.confirm('重新建立会话后，无法再访问旧会话记录。确认继续？')) return
  resetRecognitionSession()
  tasks.value = []
  total.value = 0
  error.value = ''
  expired.value = false
}
onMounted(async () => {
  void load()
  try {
    connected.value = (await fetchCapabilities())?.classifier.available ?? false
  } catch {
    connected.value = false
  }
})
</script>
<template>
  <section class="content-view">
    <PageHeading eyebrow="YOUR SPACE" title="我的" description="收藏、识别记录与设备状态。" />
    <div class="section-heading">
      <h2>
        识别记录 <span>{{ total }}</span>
      </h2>
      <button class="text-button" :disabled="loading" @click="load()">
        {{ loading ? '读取中…' : '刷新' }}
      </button>
    </div>
    <p v-if="error" class="notice" role="alert">
      {{ error
      }}<button v-if="expired" class="text-button" @click="resetSession">重新建立会话</button>
    </p>
    <div v-if="tasks.length" class="recognition-history">
      <article v-for="task in tasks" :key="task.taskId" class="history-row">
        <RouterLink :to="`/recognize?task=${task.taskId}`"
          ><strong>{{
            task.result ? `${task.result.prediction} · ${taskLabel(task)}` : taskLabel(task)
          }}</strong
          ><span>{{ new Date(task.createdAt).toLocaleString('zh-CN') }}</span></RouterLink
        >
        <button
          class="text-button"
          :disabled="!!deleting || task.status === 'queued' || task.status === 'running'"
          @click="remove(task.taskId)"
        >
          {{ deleting === task.taskId ? '删除中' : '删除' }}
        </button>
      </article>
      <button
        v-if="tasks.length < total"
        class="secondary-button"
        :disabled="loading"
        @click="load(true)"
      >
        加载更多
      </button>
    </div>
    <div v-else-if="!loading && !error" class="empty-inline">
      <p>{{ apiBase ? '本次会话还没有识别记录。' : '尚未连接识别记录服务。' }}</p>
      <RouterLink to="/recognize" class="text-button">选择图片试分类 →</RouterLink>
    </div>
    <p class="history-note">
      识别图片与结果保存在服务端，由当前会话凭据访问。网页关闭标签页后会清除凭据，请保留此页以便继续查看。
    </p>

    <div class="section-heading mine-section">
      <h2>
        我的收藏 <span>{{ atlas.saved.value.length }}</span>
      </h2>
      <span class="save-note">仅保存在本机 · 最近收藏在前</span>
    </div>
    <p v-if="atlas.missingSaved.value.length" class="storage-warning" role="status">
      另有 {{ atlas.missingSaved.value.length }} 条收藏不在当前目录中（可能来自其他目录或已删除）。
      <button class="text-button" @click="atlas.pruneSaved()">清理失效收藏</button>
    </p>
    <div v-if="atlas.saved.value.length" class="record-list">
      <SlideCard v-for="slide in atlas.saved.value" :key="slide.id" :slide="slide" />
    </div>
    <div v-else class="empty-state">
      <Bookmark :size="40" :stroke-width="1" />
      <h2>还没有收藏</h2>
      <p>在点位卡片或详情中点击收藏，方便再次查看。</p>
      <RouterLink to="/" class="secondary-button">开始探索 <ArrowRight :size="17" /></RouterLink>
    </div>

    <div class="section-heading mine-section">
      <h2>版本与服务状态</h2>
    </div>
    <div class="service-status">
      <span :class="{ ready: connected }"></span
      >{{ connected ? '单图分类服务已连接' : '识别服务未连接' }}
    </div>
    <div class="info-card">
      <Info :size="18" />
      <p>收藏保存在本机；识别时图片会上传到已配置的服务。公开影像示例共 8 张，模型结果均需复核。</p>
    </div>
  </section>
</template>
