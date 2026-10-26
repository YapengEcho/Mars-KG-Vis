import type { Slide } from '../types'
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
export async function classifyImage(file: File): Promise<unknown> {
  if (!apiBase) throw new Error('识别服务尚未接入')
  const body = new FormData()
  body.append('file', file)
  const r = await fetch(`${apiBase}/api/landslide/classify`, {
    method: 'POST',
    body,
    signal: AbortSignal.timeout(60000),
  })
  if (!r.ok) throw new Error(r.status === 503 ? '识别服务尚未就绪' : '识别失败，请稍后重试')
  return r.json()
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
