import type { Slide } from '../types'
import { chatEventData } from './chatEvents'
const configured = import.meta.env.VITE_API_BASE_URL?.trim()
export const apiBase = configured ? configured.replace(/\/$/, '') : ''
export interface Capabilities {
  classifier: {
    available: boolean
    screeningAvailable?: boolean
    reason?: string
    historyAvailable?: boolean
    modelVersion?: string
  }
  detection: {
    available: boolean
    reason?: string
    modelVersion?: string | null
    preprocessingVersion?: string | null
    maxImageSide?: number
    maxImagePixels?: number
    maxUploadBytes?: number
    maxCandidates?: number
  }
  agent?: { available: boolean; model: string; reason?: string }
}
export async function fetchCapabilities(): Promise<Capabilities | null> {
  if (!apiBase) return null
  const r = await fetch(`${apiBase}/api/capabilities`, { signal: AbortSignal.timeout(8000) })
  if (!r.ok) throw new Error('暂时无法连接识别服务')
  return r.json() as Promise<Capabilities>
}
/** Boundary geometry from the existing content API; `null` when the record has none. */
export async function fetchSlideGeometry(id: string): Promise<unknown | null> {
  if (!apiBase) return null
  const r = await fetch(`${apiBase}/api/landslides/${encodeURIComponent(id)}/geometry`, {
    signal: AbortSignal.timeout(10000),
  })
  if (r.status === 404) return null
  if (!r.ok) throw new Error('边界几何暂时不可用')
  return r.json() as Promise<unknown>
}
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}
export interface Citation {
  refType: string
  refId: string
  title: string
  url: string | null
}
export interface AgentAction {
  type: 'locate' | 'open_recognize' | 'open_task'
  slideId?: string
  taskId?: string
}
export interface ToolCallRecord {
  name: string
  args: Record<string, unknown>
  ok: boolean
}
export interface ChatResponse {
  status: 'ok' | 'llm_unavailable' | 'error'
  answer: string
  records: Slide[]
  citations: Citation[]
  actions: AgentAction[]
  toolCalls: ToolCallRecord[]
  traceId?: string
  version?: string
  /** Present only when the request selected a recognition task. */
  recognitionTask?: RecognitionTaskRef | null
}

/** Minimal task identity the chat page needs to route 「打开原记录」. */
export interface RecognitionTaskRef {
  kind: 'classify' | 'detect'
  status: string
}

/** Selected recognition task forwarded with a chat request. */
export interface ChatTaskContext {
  recognitionTaskId: string
  /** Device bearer header; the backend rejects the task without it. */
  headers: Record<string, string>
}

export interface ExternalChatContext {
  reply: ChatResponse | null
  excerpts: { title: string; text: string }[]
  citations: Citation[]
}

export async function fetchExternalChatContext(
  question: string,
): Promise<ExternalChatContext | null> {
  if (!apiBase || question.length > 4000) return null
  try {
    const response = await fetch(`${apiBase}/api/agent/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      cache: 'no-store',
      signal: AbortSignal.timeout(3000),
    })
    return response.ok ? await response.json() : null
  } catch {
    return null
  }
}
/** Task id only travels when a task is selected, so plain chat stays task-free. */
function chatTaskBody(task?: ChatTaskContext): { recognitionTaskId?: string } {
  return task ? { recognitionTaskId: task.recognitionTaskId } : {}
}

export async function sendChatMessage(
  messages: ChatMessage[],
  sessionId?: string,
  task?: ChatTaskContext,
): Promise<ChatResponse> {
  if (!apiBase) {
    return {
      status: 'llm_unavailable',
      answer: '尚未连接后端服务：请先启动 FastAPI，并在 .env.local 中配置 VITE_API_BASE_URL。',
      records: [],
      citations: [],
      actions: [],
      toolCalls: [],
    }
  }
  const r = await fetch(`${apiBase}/api/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(task?.headers ?? {}) },
    body: JSON.stringify({ sessionId: sessionId ?? null, messages, ...chatTaskBody(task) }),
    signal: AbortSignal.timeout(90000),
  })
  if (!r.ok) throw new Error('助手服务暂时不可用，请稍后重试')
  return r.json() as Promise<ChatResponse>
}

/** One SSE frame emitted by POST /api/agent/chat/stream. */
export interface ChatStreamEvent {
  type: 'delta' | 'status' | 'done' | 'error'
  content?: string
  tool?: string
  ok?: boolean
  response?: ChatResponse
}
export interface ChatStreamHandlers {
  onDelta?: (content: string) => void
  onTool?: (name: string, ok: boolean) => void
  /** Fired for both `done` and `error` events; check response.status. */
  onDone?: (response: ChatResponse) => void
}
/**
 * Streaming variant of sendChatMessage. `onDelta` fires per text chunk so the
 * UI can render the answer as it is generated; `onDone` carries the final
 * ChatResponse (records, citations, actions) and marks the end of the stream.
 */
export async function streamChatMessage(
  messages: ChatMessage[],
  handlers: ChatStreamHandlers,
  sessionId?: string,
  task?: ChatTaskContext,
): Promise<void> {
  if (!apiBase) {
    handlers.onDone?.({
      status: 'llm_unavailable',
      answer: '尚未连接后端服务：请先启动 FastAPI，并在 .env.local 中配置 VITE_API_BASE_URL。',
      records: [],
      citations: [],
      actions: [],
      toolCalls: [],
    })
    return
  }
  const r = await fetch(`${apiBase}/api/agent/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(task?.headers ?? {}) },
    body: JSON.stringify({ sessionId: sessionId ?? null, messages, ...chatTaskBody(task) }),
    signal: AbortSignal.timeout(90000),
  })
  if (!r.ok || !r.body) {
    // 选中任务时，401/404 会在 SSE 开始前以普通 JSON 返回，按错误码说明原因。
    const body = await r.json().catch(() => null)
    const code = body?.detail?.code
    if (code === 'AUTH_REQUIRED') throw new Error('识别凭据已失效，请重新打开识别记录后再问。')
    if (code === 'TASK_NOT_FOUND') throw new Error('这条记录不存在，或不属于当前设备。')
    throw new Error('助手服务暂时不可用，请稍后重试')
  }
  for await (const data of chatEventData(r.body)) {
    const event = JSON.parse(data) as ChatStreamEvent
    if (event.type === 'delta' && event.content) handlers.onDelta?.(event.content)
    else if (event.type === 'status' && event.tool) handlers.onTool?.(event.tool, event.ok ?? false)
    else if ((event.type === 'done' || event.type === 'error') && event.response) {
      handlers.onDone?.(event.response)
    }
  }
}
