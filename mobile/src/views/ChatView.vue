<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { LoaderCircle, MapPin, ScanLine, Send, Sparkles } from '@lucide/vue'
import AssistantMarkdown from '../components/AssistantMarkdown.vue'
import ChatModelSettings from '../components/ChatModelSettings.vue'
import SlideCard from '../components/SlideCard.vue'
import { isChatWebLink } from '../lib/chatMarkdown'
import { fetchCapabilities } from '../services/api'
import type { AgentAction } from '../services/api'
import { chat } from '../stores/chat'
import type { ChatEntry } from '../stores/chat'
const route = useRoute()
const router = useRouter()
const draft = ref('')
const agentReady = ref<boolean | null>(null)
const scrollEl = ref<HTMLElement>()
const thread = computed(() => chat.state.thread)
const busy = computed(() => chat.state.busy)
const error = computed(() => chat.state.error)
const modelStatus = computed(() => chat.state.model?.model ?? statusText())
// 识别/检测记录页通过 ?task=&mode= 把选中的任务交给助手。
const selectedTaskId = ref('')
const selectedTaskMode = ref<'classify' | 'detect' | ''>('')
function syncTaskFromQuery() {
  const task = typeof route.query.task === 'string' ? route.query.task : ''
  const valid = /^[a-f0-9]{32}$/.test(task)
  selectedTaskId.value = valid ? task : ''
  selectedTaskMode.value = valid ? (route.query.mode === 'detect' ? 'detect' : 'classify') : ''
}
function clearTask() {
  selectedTaskId.value = ''
  selectedTaskMode.value = ''
  router.replace({ path: '/chat' })
}
watch(() => [route.query.task, route.query.mode], syncTaskFromQuery)
onMounted(async () => {
  syncTaskFromQuery()
  await scrollToEnd()
  try {
    const capabilities = await fetchCapabilities()
    agentReady.value = capabilities?.agent?.available ?? null
  } catch {
    agentReady.value = null
  }
})
async function scrollToEnd() {
  await nextTick()
  const element = scrollEl.value
  if (element) element.scrollTop = element.scrollHeight
}
async function send() {
  const text = draft.value.trim()
  if (!text || chat.state.busy) return
  draft.value = ''
  await scrollToEnd()
  await chat.send(text, selectedTaskId.value || undefined)
  await scrollToEnd()
}
function resetThread() {
  if (chat.state.busy) return
  chat.reset()
  draft.value = ''
  scrollToEnd()
}
function runAction(action: AgentAction, message?: ChatEntry) {
  if (action.type === 'locate' && action.slideId) {
    router.push({ path: '/', query: { focus: action.slideId } })
  } else if (action.type === 'open_recognize') {
    router.push('/recognize')
  } else if (action.type === 'open_task' && action.taskId) {
    const taskKind = message?.taskKind ?? selectedTaskMode.value
    router.push({
      path: '/recognize',
      query: { task: action.taskId, ...(taskKind === 'detect' ? { mode: 'detect' } : {}) },
    })
  }
}
function statusText() {
  if (agentReady.value === true) return '内置助手已连接'
  if (agentReady.value === false) return '内置模型未配置'
  return '内置助手未连接'
}
</script>
<template>
  <section class="chat-shell">
    <div class="chat-top">
      <div class="chat-title">
        <span class="eyebrow">MARS ASSISTANT</span>
        <h1>滑坡助手</h1>
        <p class="chat-model-status" :title="modelStatus">{{ modelStatus }}</p>
      </div>
      <div class="chat-actions">
        <ChatModelSettings />
        <button class="text-button" type="button" :disabled="busy" @click="resetThread">
          新建对话
        </button>
      </div>
    </div>
    <div ref="scrollEl" class="chat-thread">
      <template v-for="(message, index) in thread" :key="index">
        <div class="bubble" :class="message.role">
          <AssistantMarkdown v-if="message.role === 'assistant'" :content="message.content" />
          <p v-else>{{ message.content }}</p>
        </div>
        <div v-if="message.citations?.length" class="citation-list">
          <a
            v-for="citation in message.citations"
            :key="citation.refId"
            :href="isChatWebLink(citation.url) ? citation.url : undefined"
            :class="{ disabled: !isChatWebLink(citation.url) }"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Sparkles :size="13" />{{ citation.title }}
          </a>
        </div>
        <div v-if="message.records?.length" class="chat-records">
          <SlideCard v-for="record in message.records" :key="record.id" :slide="record" />
        </div>
        <div v-if="message.actions?.length" class="action-row">
          <button
            v-for="(action, actionIndex) in message.actions"
            :key="actionIndex"
            class="action-chip"
            @click="runAction(action, message)"
          >
            <MapPin v-if="action.type === 'locate'" :size="14" />
            <ScanLine v-else :size="14" />
            {{
              action.type === 'locate'
                ? '在地图上定位'
                : action.type === 'open_recognize'
                  ? '去识别'
                  : '打开任务'
            }}
          </button>
        </div>
      </template>
      <div
        v-if="busy && thread[thread.length - 1]?.role === 'user'"
        class="bubble assistant typing"
      >
        <LoaderCircle :size="16" class="spin" />
      </div>
    </div>
    <p v-if="error" class="chat-error" role="alert">{{ error }}</p>
    <div v-if="selectedTaskId" class="task-chip">
      <ScanLine :size="13" />
      <span>
        正在解释选中的{{ selectedTaskMode === 'detect' ? '检测' : '识别' }}记录 ·
        {{ selectedTaskId.slice(0, 8) }}
      </span>
      <button type="button" class="task-chip-clear" :disabled="busy" @click="clearTask">
        取消
      </button>
    </div>
    <form class="chat-input" @submit.prevent="send">
      <input
        v-model="draft"
        placeholder="问问滑坡目录、统计、类型资料……"
        aria-label="向助手提问"
        :disabled="busy"
      />
      <button type="submit" class="send-btn" :disabled="!draft.trim() || busy" aria-label="发送">
        <Send :size="17" />
      </button>
    </form>
  </section>
</template>
<style scoped>
.chat-shell {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 850px;
  margin: 0 auto;
  height: calc(100dvh - 100px - 108px);
  padding: 22px 6% 0;
}
.chat-top {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 14px;
}
.chat-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
.chat-actions .text-button:disabled {
  color: var(--muted);
  cursor: not-allowed;
}
.chat-top h1 {
  font-size: 26px;
  font-weight: 450;
  margin: 6px 0 0;
}
.chat-title {
  min-width: 0;
}
.chat-model-status {
  margin-top: 4px;
  overflow: hidden;
  color: var(--muted);
  font-size: var(--text-xs);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chat-thread {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 2px 12px;
}
.bubble {
  max-width: 84%;
  padding: 11px 14px;
  border-radius: 14px;
}
.bubble p {
  font-size: 13px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.bubble.user {
  align-self: flex-end;
  background: var(--accent);
  color: #091321;
  border-bottom-right-radius: 5px;
}
.bubble.assistant {
  align-self: flex-start;
  background: var(--surface);
  border: 1px solid var(--line);
  border-bottom-left-radius: 5px;
}
.typing {
  display: flex;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.citation-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  align-self: flex-start;
  max-width: 84%;
  padding-left: 6px;
}
.citation-list a {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  font-size: 11px;
  color: var(--accent);
}
.citation-list a.disabled {
  color: var(--muted);
  pointer-events: none;
}
.chat-records {
  display: grid;
  gap: 10px;
  align-self: stretch;
}
.action-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-self: flex-start;
}
.action-chip {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 8px 13px;
  border: 1px solid var(--selected-border);
  border-radius: 20px;
  background: var(--selected-bg);
  font-size: 11px;
  color: var(--text);
}
.chat-error {
  font-size: 12px;
  color: #e8a99f;
  margin: 0 0 8px;
}
.task-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 8px;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  font-size: 11px;
  color: var(--muted);
}
.task-chip span {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.task-chip-clear {
  border: none;
  background: none;
  color: var(--accent);
  font-size: 11px;
  cursor: pointer;
  padding: 0;
}
.task-chip-clear:disabled {
  opacity: 0.5;
  cursor: default;
}
.chat-input {
  display: flex;
  gap: 10px;
  align-items: center;
  padding-bottom: 12px;
}
.chat-input input {
  flex: 1;
  min-width: 0;
  height: 46px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 23px;
  padding: 0 18px;
  color: var(--text);
  font-size: 13px;
  outline: none;
}
.chat-input input:focus {
  border-color: var(--accent);
}
.send-btn {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  background: var(--accent);
  color: #091321;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
@media (max-width: 600px) {
  .chat-shell {
    height: calc(100dvh - 83px - 108px);
    padding: 16px 22px 0;
  }
}
</style>
