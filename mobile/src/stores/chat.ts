import { reactive } from 'vue'
import { buildChatHistory } from '../lib/chatHistory'
import { recognitionAuthHeaders } from '../services/recognition'
import { streamChatMessage } from '../services/api'
import type { AgentAction, ChatResponse, ChatStreamHandlers } from '../services/api'
import { prepareExternalModel, streamExternalChat } from '../services/externalChat'
import type { ExternalModel, ExternalModelSettings } from '../services/externalChat'

export const WELCOME_MESSAGE =
  '你好，我是火星滑坡助手。可以问我：某类滑坡有哪些、数量统计、类型与案例资料，或让我把某个点位定位到地图上。'

export interface ChatEntry {
  role: 'user' | 'assistant'
  content: string
  records?: ChatResponse['records']
  citations?: ChatResponse['citations']
  actions?: AgentAction[]
  status?: ChatResponse['status']
  /** Task kind of the explained record, so 「打开原记录」 can pick the right mode. */
  taskKind?: 'classify' | 'detect'
  /**
   * Assistant replies stay pending until a terminal `done`/`error` frame
   * arrives. Pending content is shown to the user but excluded from the history
   * sent to the model, so a cut-off answer is never replayed as complete.
   */
  pending?: boolean
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
  model: null as ExternalModelSettings | null,
})

// The credential stays in memory and is separate from the UI and transcripts.
let externalModel: ExternalModel | null = null

function reset() {
  state.thread = [{ role: 'assistant', content: WELCOME_MESSAGE }]
  state.error = ''
}

/**
 * Credential for a chat request that explains a selected recognition task.
 *
 * The device bearer header is required by the backend; without an existing
 * session we refuse to send rather than silently create a new device identity.
 */
async function taskContext(
  taskId: string,
): Promise<{ recognitionTaskId: string; headers: Record<string, string> } | null> {
  let headers: Record<string, string> | null = null
  try {
    headers = await recognitionAuthHeaders()
  } catch {
    headers = null
  }
  return headers ? { recognitionTaskId: taskId, headers } : null
}

export const chat = {
  state,
  reset,
  setModel(draft: ExternalModel | null) {
    if (state.busy) return
    const next = draft ? prepareExternalModel(draft, externalModel) : null
    if (
      externalModel?.provider === next?.provider &&
      externalModel?.endpoint === next?.endpoint &&
      externalModel?.model === next?.model &&
      externalModel?.apiKey === next?.apiKey
    )
      return
    externalModel = next
    state.model = next
      ? { provider: next.provider, endpoint: next.endpoint, model: next.model }
      : null
    reset()
  },
  async send(text: string, taskId?: string): Promise<boolean> {
    const question = text.trim()
    if (!question || state.busy) return false
    state.error = ''
    // 选中识别任务时必须走内置助手：外部模型读不到服务端的任务与凭据。
    if (taskId && externalModel) {
      state.error = '解释识别记录需要使用内置助手，请先关闭自定义模型。'
      return false
    }
    let task: { recognitionTaskId: string; headers: Record<string, string> } | undefined
    if (taskId) {
      const context = await taskContext(taskId)
      if (!context) {
        state.error = '请在本设备的识别记录中打开任务，再让助手解释。'
        return false
      }
      task = context
    }
    state.thread.push({ role: 'user', content: question })
    state.busy = true
    // The assistant bubble is created lazily on the first streamed chunk and
    // grows as deltas arrive; onDone swaps in the authoritative final payload
    // and clears `pending`. If the stream dies early the bubble stays pending
    // and is skipped when building the next request's history.
    let entryIndex = -1
    let finished = false
    const ensureEntry = (): ChatEntry => {
      if (entryIndex < 0) {
        state.thread.push({ role: 'assistant', content: '', pending: true })
        entryIndex = state.thread.length - 1
      }
      return state.thread[entryIndex]
    }
    try {
      const messages = buildChatHistory(state.thread)
      const handlers: ChatStreamHandlers = {
        onDelta: (chunk) => {
          ensureEntry().content += chunk
        },
        onDone: (response) => {
          finished = true
          const entry = ensureEntry()
          entry.content = response.answer
          entry.records = response.records
          entry.citations = response.citations
          entry.actions = response.actions
          entry.status = response.status
          entry.taskKind = response.recognitionTask?.kind
          entry.pending = false
        },
      }
      if (externalModel) await streamExternalChat(externalModel, messages, handlers)
      else await streamChatMessage(messages, handlers, undefined, task)
      if (!finished) state.error = '回复未完成，请重试'
    } catch (e) {
      state.error = e instanceof Error ? e.message : '发送失败，请稍后重试'
    } finally {
      state.busy = false
    }
    return true
  },
}
