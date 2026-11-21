<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  ArrowRight,
  Bookmark,
  ChevronDown,
  ChevronRight,
  History,
  RefreshCw,
  Server,
  Trash2,
} from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import { atlas } from '../stores/atlas'
import { apiBase, fetchCapabilities } from '../services/api'
import { typeInfo } from '../types'
import {
  recognitionHistory,
  deleteRecognition,
  taskLabel,
  resetRecognitionSession,
  RecognitionError,
  type HistoryKind,
  type HistoryTask,
} from '../services/recognition'
const tasks = ref<HistoryTask[]>([])
const kind = ref<HistoryKind>('classify')
const total = ref(0)
const page = ref(1)
const loading = ref(false)
const error = ref('')
const expired = ref(false)
const deleting = ref('')
const service = ref<'checking' | 'connected' | 'offline'>(apiBase ? 'checking' : 'offline')
const serviceLabel = { checking: '检查中', connected: '已连接', offline: '未连接' }

async function load(more = false) {
  if (loading.value) return
  loading.value = true
  error.value = ''
  expired.value = false
  try {
    const next = more ? page.value + 1 : 1
    const result = await recognitionHistory(next, kind.value)
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
  if (
    loading.value ||
    deleting.value ||
    !window.confirm('删除这条识别记录和预览图片？此操作无法恢复。')
  )
    return
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
  page.value = 1
  error.value = ''
  expired.value = false
}
function switchKind(next: HistoryKind) {
  if (kind.value === next) return
  kind.value = next
  tasks.value = []
  total.value = 0
  page.value = 1
  error.value = ''
  expired.value = false
  void load()
}
onMounted(async () => {
  void load()
  try {
    const classifier = (await fetchCapabilities())?.classifier
    service.value = classifier?.available && classifier.screeningAvailable ? 'connected' : 'offline'
  } catch {
    service.value = 'offline'
  }
})
</script>
<template>
  <section class="content-view mine-view">
    <header class="mine-heading">
      <h1>我的</h1>
    </header>

    <div class="mine-grid">
      <section class="mine-panel" aria-labelledby="mine-history-title">
        <header class="mine-panel-heading">
          <div class="mine-panel-title">
            <span class="mine-panel-icon"><History :size="18" aria-hidden="true" /></span>
            <h2 id="mine-history-title">识别记录</h2>
            <span v-if="tasks.length || (!loading && !error)" class="mine-count"
              >{{ total }} 条</span
            >
          </div>
          <button
            class="mine-action"
            type="button"
            aria-label="刷新识别记录"
            :disabled="loading || !!deleting"
            @click="load()"
          >
            <RefreshCw :size="15" aria-hidden="true" />刷新
          </button>
        </header>
        <div class="mine-kind-tabs" role="tablist" aria-label="记录类型">
          <button
            type="button"
            role="tab"
            :aria-selected="kind === 'classify'"
            :class="{ active: kind === 'classify' }"
            @click="switchKind('classify')"
          >
            单图分类
          </button>
          <button
            type="button"
            role="tab"
            :aria-selected="kind === 'detect'"
            :class="{ active: kind === 'detect' }"
            @click="switchKind('detect')"
          >
            裁图检测
          </button>
        </div>

        <div v-if="error" class="mine-message">
          <p role="alert">{{ error }}</p>
          <button v-if="expired" class="mine-action" type="button" @click="resetSession">
            重新建立会话
          </button>
        </div>
        <ul v-if="tasks.length" class="mine-list" :aria-busy="loading">
          <li v-for="task in tasks" :key="task.taskId" class="mine-history-row">
            <RouterLink
              :to="
                task.kind === 'detect'
                  ? { path: '/recognize', query: { mode: 'detect', task: task.taskId } }
                  : { path: '/recognize', query: { task: task.taskId } }
              "
              class="mine-history-link"
            >
              <div class="mine-history-copy">
                <div class="mine-history-title">
                  <strong>{{ taskLabel(task) }}</strong>
                  <span
                    v-if="
                      task.kind === 'classify' &&
                      task.status === 'completed' &&
                      task.screening?.decision === 'possible_landslide' &&
                      task.result
                    "
                    >{{ typeInfo[task.result.prediction]?.name ?? task.result.prediction }}</span
                  >
                </div>
                <time :datetime="task.createdAt">{{
                  new Date(task.createdAt).toLocaleString('zh-CN')
                }}</time>
              </div>
              <ChevronRight :size="16" aria-hidden="true" />
            </RouterLink>
            <button
              class="mine-action mine-delete"
              type="button"
              :aria-label="deleting === task.taskId ? '正在删除识别记录' : '删除识别记录'"
              :disabled="
                loading || !!deleting || task.status === 'queued' || task.status === 'running'
              "
              @click="remove(task.taskId)"
            >
              <Trash2 :size="17" aria-hidden="true" />
            </button>
          </li>
        </ul>
        <div v-if="tasks.length && tasks.length < total" class="mine-more">
          <button
            class="mine-action"
            type="button"
            :disabled="loading || !!deleting"
            @click="load(true)"
          >
            {{ loading ? '读取中…' : '加载更多' }}
          </button>
        </div>
        <p v-if="loading && !tasks.length" class="mine-message" role="status">正在读取识别记录…</p>
        <div v-else-if="!tasks.length && !error" class="mine-empty">
          <h3>{{ apiBase ? (kind === 'detect' ? '还没有检测记录' : '还没有识别记录') : '识别记录服务未配置' }}</h3>
          <p v-if="apiBase">
            {{
              kind === 'detect'
                ? '上传整幅 CTX 裁图后，可在这里查看候选框结果。'
                : '上传 CTX 裁片后，可在这里查看结果。'
            }}
          </p>
          <RouterLink
            :to="kind === 'detect' ? { path: '/recognize', query: { mode: 'detect' } } : '/recognize'"
            class="mine-entry"
          >
            {{ kind === 'detect' ? '前往检测' : '前往识别' }}
            <ArrowRight :size="16" aria-hidden="true" />
          </RouterLink>
        </div>
      </section>

      <section class="mine-panel" aria-labelledby="mine-saved-title">
        <header class="mine-panel-heading">
          <div class="mine-panel-title">
            <span class="mine-panel-icon"><Bookmark :size="18" aria-hidden="true" /></span>
            <h2 id="mine-saved-title">收藏</h2>
            <span class="mine-count">{{ atlas.saved.value.length }} 条</span>
          </div>
          <span class="mine-local-note">本机保存</span>
        </header>
        <div
          v-if="!atlas.state.loading && !atlas.state.error && atlas.missingSaved.value.length"
          class="mine-message"
        >
          <p role="status">有 {{ atlas.missingSaved.value.length }} 条收藏未在当前目录中显示。</p>
          <button class="mine-action" type="button" @click="atlas.pruneSaved()">
            清理失效收藏
          </button>
        </div>
        <ul v-if="atlas.saved.value.length" class="mine-list mine-saved-list">
          <li v-for="slide in atlas.saved.value" :key="slide.id">
            <SlideCard :slide="slide" />
          </li>
        </ul>
        <div v-else class="mine-empty">
          <h3>还没有收藏</h3>
          <p>打开滑坡详情即可收藏。</p>
          <RouterLink to="/" class="mine-entry">
            探索滑坡 <ArrowRight :size="16" aria-hidden="true" />
          </RouterLink>
        </div>
      </section>
    </div>

    <section class="mine-service" aria-labelledby="mine-service-title">
      <div class="mine-service-row">
        <h2 id="mine-service-title"><Server :size="16" aria-hidden="true" />单图识别服务</h2>
        <p
          class="mine-service-state"
          :class="{ 'is-connected': service === 'connected' }"
          role="status"
        >
          {{ serviceLabel[service] }}
        </p>
      </div>
      <details class="mine-policy">
        <summary>识别记录保存说明 <ChevronDown :size="16" aria-hidden="true" /></summary>
        <p>识别图片和结果保存在服务端，仅凭本次会话访问。关闭标签页会清除浏览器的会话凭据。</p>
      </details>
    </section>
  </section>
</template>

<style scoped>
.mine-kind-tabs {
  display: inline-flex;
  gap: 2px;
  margin: 0 0 var(--space-sm);
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: 999px;
}
.mine-kind-tabs button {
  padding: 0.3em 0.8em;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: var(--text-sm);
  cursor: pointer;
}
.mine-kind-tabs button.active {
  background: color-mix(in srgb, var(--accent-hover) 16%, transparent);
  color: inherit;
}
.mine-view {
  --mine-padding: clamp(14px, 2vw, 22px);
  --mine-gap: clamp(14px, 2.5vw, 24px);
  max-width: 1032px;
}
.mine-heading {
  margin-block-end: var(--mine-gap);
}
.mine-heading h1 {
  font-size: clamp(1.75rem, 3.2vw, 2.375rem);
  font-weight: 650;
  line-height: 1.15;
}
.mine-grid {
  display: grid;
  gap: var(--mine-gap);
  align-items: start;
}
.mine-panel,
.mine-service {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
}
.mine-panel {
  background: var(--surface);
}
.mine-panel-heading,
.mine-panel-title,
.mine-action,
.mine-entry,
.mine-service-row,
.mine-service-row h2,
.mine-policy summary {
  display: flex;
  align-items: center;
}
.mine-panel-heading {
  justify-content: space-between;
  gap: var(--space-2xs);
  min-height: 66px;
  padding: var(--space-xs) var(--mine-padding);
  border-bottom: 1px solid var(--line);
}
.mine-panel-title {
  flex-wrap: wrap;
  gap: var(--space-xs);
}
.mine-panel-icon {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: var(--selected-bg);
  color: var(--accent-hover);
}
.mine-panel h2 {
  font-size: var(--text-lg);
  font-weight: 650;
}
.mine-count,
.mine-local-note {
  color: var(--muted);
  font-size: var(--text-sm);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.mine-action,
.mine-entry {
  width: fit-content;
  min-height: 44px;
  gap: var(--space-2xs);
  font-size: var(--text-sm);
}
.mine-action {
  justify-content: center;
  padding-inline: var(--space-2xs);
  color: var(--text-secondary);
}
.mine-action:enabled:hover,
.mine-policy summary:hover {
  color: var(--accent-hover);
}
.mine-entry {
  gap: var(--space-xs);
  margin-block-start: var(--space-2xs);
  padding-inline: var(--space-sm);
  border: 1px solid color-mix(in oklab, var(--accent) 38%, var(--line));
  border-radius: 6px;
  color: var(--accent-hover);
}
.mine-entry:hover {
  background: var(--selected-bg);
  border-color: var(--accent);
}
.mine-empty {
  display: grid;
  align-content: center;
  gap: var(--space-2xs);
  min-height: 148px;
  padding: var(--space-md) var(--mine-padding);
}
.mine-empty h3 {
  font-size: var(--text-md);
  font-weight: 600;
}
.mine-empty p,
.mine-message {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
.mine-message {
  padding: var(--space-sm) var(--mine-padding);
}
.mine-message:has(+ .mine-list),
.mine-message:has(+ .mine-empty) {
  border-bottom: 1px solid var(--line);
}
.mine-message .mine-action {
  margin-inline-start: calc(-1 * var(--space-2xs));
}
.mine-list {
  margin: 0;
  padding: 0 var(--mine-padding);
  list-style: none;
}
.mine-list li + li,
.mine-more {
  border-top: 1px solid var(--line);
}
.mine-history-row,
.mine-saved-list :deep(.slide-card) {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--space-xs);
}
.mine-history-link,
.mine-saved-list :deep(.slide-card-main) {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-xs);
  min-height: 68px;
  padding-block: var(--space-xs);
  border-radius: 6px;
}
.mine-history-link:hover,
.mine-saved-list :deep(.slide-card-main:hover) {
  color: var(--accent-hover);
}
.mine-history-link > svg {
  color: var(--muted);
}
.mine-history-copy {
  min-width: 0;
}
.mine-history-title {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: var(--space-2xs) var(--space-xs);
}
.mine-history-title strong {
  font-size: var(--text-sm);
  font-weight: 600;
}
.mine-history-title span,
.mine-history-copy time {
  color: var(--muted);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}
.mine-delete,
.mine-saved-list :deep(.save-icon) {
  width: 44px;
  min-height: 44px;
  padding: 0;
  border-radius: 6px;
}
.mine-delete:enabled:hover {
  color: var(--color-error);
  background: var(--surface-raised);
}
.mine-more {
  display: flex;
  justify-content: center;
  margin-inline: var(--mine-padding);
}
.mine-saved-list :deep(.slide-card) {
  padding: 0;
  border: 0;
  background: none;
}
.mine-saved-list :deep(.slide-thumbnail) {
  flex-basis: 42px;
  width: 42px;
  height: 42px;
  border-radius: 4px;
}
.mine-saved-list :deep(.slide-card h3) {
  margin-block-end: var(--space-3xs);
  font-size: var(--text-sm);
}
.mine-saved-list :deep(.card-facts) {
  font-size: var(--text-xs);
}
.mine-saved-list :deep(.save-icon) {
  position: static;
  color: var(--accent-hover);
  background: transparent;
}
.mine-saved-list :deep(.save-icon:hover) {
  background: var(--surface-raised);
}
.mine-service {
  margin-block-start: var(--mine-gap);
  padding-inline: var(--mine-padding);
}
.mine-service-row,
.mine-policy summary {
  justify-content: space-between;
  gap: var(--space-xs);
  min-height: 48px;
}
.mine-service-row h2,
.mine-service-state,
.mine-policy {
  font-size: var(--text-sm);
}
.mine-service-row h2 {
  gap: var(--space-xs);
  font-weight: 600;
}
.mine-service-row h2 svg {
  color: var(--muted);
}
.mine-service-state {
  display: flex;
  align-items: center;
  gap: var(--space-2xs);
  color: var(--text-secondary);
  white-space: nowrap;
}
.mine-service-state::before {
  content: '';
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted);
}
.mine-service-state.is-connected::before {
  background: var(--color-retained-15);
}
.mine-policy {
  border-top: 1px solid var(--line);
  color: var(--text-secondary);
}
.mine-policy summary {
  cursor: pointer;
  list-style: none;
}
.mine-policy summary::-webkit-details-marker {
  display: none;
}
.mine-policy[open] svg {
  rotate: 180deg;
}
.mine-policy p {
  padding-block-end: var(--space-md);
  color: var(--muted);
  line-height: 1.6;
}

@media (min-width: 800px) {
  .mine-grid {
    grid-template-columns: minmax(0, 1.12fr) minmax(0, 1fr);
  }
}
</style>
