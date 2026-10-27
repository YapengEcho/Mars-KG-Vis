<script setup lang="ts">
import { nextTick, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { LoaderCircle, MapPin, ScanLine, Send, Sparkles } from '@lucide/vue'
import SlideCard from '../components/SlideCard.vue'
import { fetchCapabilities, sendChatMessage } from '../services/api'
import type { AgentAction, ChatResponse } from '../services/api'
import { buildChatHistory } from '../lib/chatHistory'
interface UIMessage {
  role: 'user' | 'assistant'
  content: string
  records?: ChatResponse['records']
  citations?: ChatResponse['citations']
  actions?: AgentAction[]
  status?: ChatResponse['status']
}
const router = useRouter()
const WELCOME_MESSAGE =
  '你好，我是火星滑坡助手。可以问我：某类滑坡有哪些、数量统计、类型与案例资料，或让我把某个点位定位到地图上。'
const thread = ref<UIMessage[]>([{ role: 'assistant', content: WELCOME_MESSAGE }])
const draft = ref('')
const busy = ref(false)
const error = ref('')
const agentReady = ref<boolean | null>(null)
const scrollEl = ref<HTMLElement>()
onMounted(async () => {
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
  if (!text || busy.value) return
  error.value = ''
  thread.value.push({ role: 'user', content: text })
  draft.value = ''
  busy.value = true
  await scrollToEnd()
  try {
    const history = buildChatHistory(thread.value)
    const response = await sendChatMessage(history)
    thread.value.push({
      role: 'assistant',
      content: response.answer,
      records: response.records,
      citations: response.citations,
      actions: response.actions,
      status: response.status,
    })
  } catch (e) {
    error.value = e instanceof Error ? e.message : '发送失败，请稍后重试'
  } finally {
    busy.value = false
    await scrollToEnd()
  }
}
function resetThread() {
  if (busy.value) return
  thread.value = [{ role: 'assistant', content: WELCOME_MESSAGE }]
  draft.value = ''
  error.value = ''
  scrollToEnd()
}
function runAction(action: AgentAction) {
  if (action.type === 'locate' && action.slideId) {
    router.push({ path: '/', query: { focus: action.slideId } })
  } else if (action.type === 'open_recognize') {
    router.push('/recognize')
  } else if (action.type === 'open_task' && action.taskId) {
    router.push({ path: '/recognize', query: { task: action.taskId } })
  }
}
function statusText() {
  if (agentReady.value === true) return '助手已连接'
  if (agentReady.value === false) return '助手语言模型未配置'
  return '后端未连接'
}
</script>
<template>
  <section class="chat-shell">
    <div class="chat-top">
      <div>
        <span class="eyebrow">MARS ASSISTANT</span>
        <h1>滑坡助手</h1>
      </div>
      <div class="chat-actions">
        <div class="service-status" :class="{ ready: agentReady === true }">
          <span></span>{{ statusText() }}
        </div>
        <button class="text-button" type="button" :disabled="busy" @click="resetThread">
          新建对话
        </button>
      </div>
    </div>
    <div ref="scrollEl" class="chat-thread">
      <template v-for="(message, index) in thread" :key="index">
        <div class="bubble" :class="message.role"><p>{{ message.content }}</p></div>
        <div v-if="message.citations?.length" class="citation-list">
          <a
            v-for="citation in message.citations"
            :key="citation.refId"
            :href="citation.url ?? undefined"
            :class="{ disabled: !citation.url }"
            target="_blank"
            rel="noopener"
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
            @click="runAction(action)"
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
      <div v-if="busy" class="bubble assistant typing"><LoaderCircle :size="16" class="spin" /></div>
    </div>
    <p v-if="error" class="chat-error" role="alert">{{ error }}</p>
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
.chat-top .service-status {
  margin: 0;
  padding: 8px 12px;
  font-size: 11px;
  white-space: nowrap;
}
.chat-top .service-status.ready span {
  background: #70c6ad;
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
