import type { Slide } from '../types'
import { parseClassification, type ClassificationResult } from '../lib/recognition'
const configured = import.meta.env.VITE_API_BASE_URL?.trim()
export const apiBase = configured ? configured.replace(/\/$/, '') : ''
export interface Capabilities {
  classifier: { available: boolean; reason?: string }
  detection: { available: boolean; reason?: string }
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
export class RecognitionError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message)
  }
}
function describeStatus(status: number): string {
  switch (status) {
    case 413:
      return '图片分辨率过大，请上传不超过 4096×4096 的裁图。'
    case 415:
      return '暂不支持该图片格式，请使用单帧 PNG 或 JPEG。'
    case 422:
      return '图片无法解析，请重新选择有效的裁图。'
    case 429:
      return '识别请求过多，请稍后再试。'
    case 503:
      return '识别服务尚未就绪（503），请稍后重试。'
    default:
      return `识别失败（HTTP ${status}），请稍后重试。`
  }
}
/**
 * Single-crop classification. Resolves with a validated result; rejects with a
 * RecognitionError carrying the HTTP status when the service answered.
 * `signal` lets the page cancel the wait without touching the shared timeout.
 */
export async function classifyImage(
  file: File,
  signal?: AbortSignal,
): Promise<ClassificationResult> {
  if (!apiBase)
    throw new RecognitionError('尚未连接识别服务：请先启动 FastAPI，并配置 VITE_API_BASE_URL。')
  const body = new FormData()
  body.append('file', file)
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, 65000)
  const cancel = () => controller.abort()
  signal?.addEventListener('abort', cancel)
  let response: Response
  try {
    response = await fetch(`${apiBase}/api/landslide/classify`, {
      method: 'POST',
      body,
      signal: controller.signal,
    })
  } catch {
    if (timedOut) throw new RecognitionError('识别超时（65 秒），请重试或换一张更小的裁图。')
    if (signal?.aborted) throw new RecognitionError('已取消本次识别。')
    throw new RecognitionError('无法连接识别服务，请检查网络或服务状态。')
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
  if (!response.ok) throw new RecognitionError(describeStatus(response.status), response.status)
  const parsed = parseClassification(await response.json().catch(() => null))
  if (!parsed) throw new RecognitionError('识别服务返回的数据无法识别，请稍后重试。')
  return parsed
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
}
export async function sendChatMessage(
  messages: ChatMessage[],
  sessionId?: string
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: sessionId ?? null, messages }),
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
  sessionId?: string
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: sessionId ?? null, messages }),
    signal: AbortSignal.timeout(90000),
  })
  if (!r.ok || !r.body) throw new Error('助手服务暂时不可用，请稍后重试')
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let cut: number
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('')
      if (!data) continue
      const event = JSON.parse(data) as ChatStreamEvent
      if (event.type === 'delta' && event.content) handlers.onDelta?.(event.content)
      else if (event.type === 'status' && event.tool) handlers.onTool?.(event.tool, event.ok ?? false)
      else if ((event.type === 'done' || event.type === 'error') && event.response) {
        handlers.onDone?.(event.response)
      }
    }
  }
}
