import { reactive } from 'vue'
import { buildChatHistory } from '../lib/chatHistory'
import { sendChatMessage } from '../services/api'
import type { AgentAction, ChatResponse } from '../services/api'

export const WELCOME_MESSAGE =
  '你好，我是火星滑坡助手。可以问我：某类滑坡有哪些、数量统计、类型与案例资料，或让我把某个点位定位到地图上。'

export interface ChatEntry {
  role: 'user' | 'assistant'
  content: string
  records?: ChatResponse['records']
  citations?: ChatResponse['citations']
  actions?: AgentAction[]
  status?: ChatResponse['status']
}

/**
 * Conversation state lives at module scope so that leaving the chat route and
 * coming back keeps the transcript. When it lived inside the component, every
 * route change destroyed it and the page reopened on a blank welcome message.
 */
const state = reactive({
  thread: [{ role: 'assistant', content: WELCOME_MESSAGE }] as ChatEntry[],
  busy: false,
  error: '',
})

export const chat = {
  state,
  reset() {
    state.thread = [{ role: 'assistant', content: WELCOME_MESSAGE }]
    state.error = ''
  },
  async send(text: string): Promise<boolean> {
    const question = text.trim()
    if (!question || state.busy) return false
    state.error = ''
    state.thread.push({ role: 'user', content: question })
    state.busy = true
    try {
      const response = await sendChatMessage(buildChatHistory(state.thread))
      state.thread.push({
        role: 'assistant',
        content: response.answer,
        records: response.records,
        citations: response.citations,
        actions: response.actions,
        status: response.status,
      })
    } catch (e) {
      state.error = e instanceof Error ? e.message : '发送失败，请稍后重试'
    } finally {
      state.busy = false
    }
    return true
  },
}
