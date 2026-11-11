import { apiBase } from './api'
import type { SlideType } from '../types'

export interface RecognitionResult {
  prediction: SlideType
  probabilities: Record<SlideType, number>
  confidence: number
  uncertain: boolean
  reviewReasons: string[]
  scoresCalibrated: boolean
  modelVersion: string
  preprocessingVersion: string
}
export interface RecognitionTask {
  taskId: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  createdAt: string
  finishedAt: string | null
  modelVersion: string
  preprocessingVersion: string
  result: RecognitionResult | null
  errorCode: string | null
}
interface Session {
  token: string
  expiresAt: string
}
const sessionKey = `mars-atlas:recognition-session:v1:${apiBase}`
let issuing: Promise<Session> | null = null
const messages: Record<string, string> = {
  AUTH_REQUIRED: '本次会话凭据已失效。旧记录不会被删除；重新建立会话后只能查看新记录。',
  QUEUE_FULL: '服务正在处理其他图片，请稍后再试。',
  TASK_NOT_FOUND: '这条记录不存在，或不属于当前会话。',
  TASK_BUSY: '图片仍在处理中，完成后才能删除。',
  MODEL_NOT_CONFIGURED: '识别模型尚未就绪，请稍后重试。',
  RECOGNITION_NOT_CONFIGURED: '识别记录服务尚未就绪。',
  WORKER_UNAVAILABLE: '识别服务暂时不可用。',
  INVALID_IMAGE: '图片无法读取，请重新选择 PNG 或 JPEG。',
  INVALID_ROI: '图片中的目标区域无效。',
  UPLOAD_TOO_LARGE: '图片不能超过 8 MB。',
  IMAGE_DIMENSIONS_EXCEEDED: '图片尺寸过大，请先裁出单个滑坡。',
  UNSUPPORTED_IMAGE: '只支持 PNG 或 JPEG 图片。',
  ANIMATED_IMAGE: '请选择单帧图片。',
}
export class RecognitionError extends Error {
  constructor(public code: string) {
    super(messages[code] ?? '服务请求失败，请稍后重试。')
  }
}
function savedSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(sessionKey)
    if (!raw) return null
    const value = JSON.parse(raw) as Session
    if (!value.token || !Number.isFinite(Date.parse(value.expiresAt))) throw new Error()
    if (Date.parse(value.expiresAt) <= Date.now()) throw new RecognitionError('AUTH_REQUIRED')
    return value
  } catch (e) {
    if (e instanceof RecognitionError) throw e
    throw new Error('无法读取会话凭据，请允许当前页面使用会话存储。')
  }
}
export function resetRecognitionSession() {
  sessionStorage.removeItem(sessionKey)
}
async function check(response: Response): Promise<Response> {
  if (response.ok) return response
  const body = await response.json().catch(() => null)
  throw new RecognitionError(
    response.status === 401 ? 'AUTH_REQUIRED' : (body?.detail?.code ?? 'REQUEST_FAILED'),
  )
}
async function session(create: boolean): Promise<Session | null> {
  if (!apiBase) throw new Error('尚未连接识别服务。')
  const saved = savedSession()
  if (saved || !create) return saved
  if (!issuing)
    issuing = (async () => {
      const response = await check(
        await fetch(`${apiBase}/api/recognition/sessions`, {
          method: 'POST',
          signal: AbortSignal.timeout(10000),
          cache: 'no-store',
        }),
      )
      const value = (await response.json()) as Session
      if (!value.token || !Number.isFinite(Date.parse(value.expiresAt)))
        throw new Error('会话建立失败。')
      // Do not submit a picture unless the credential is safely retained for polling/history.
      try {
        sessionStorage.setItem(sessionKey, JSON.stringify(value))
      } catch {
        throw new Error('无法保存会话凭据，图片未提交。请允许当前页面使用会话存储。')
      }
      return value
    })().finally(() => {
      issuing = null
    })
  return issuing
}
async function request(path: string, options: RequestInit = {}, create = false): Promise<Response> {
  const auth = await session(create)
  if (!auth) throw new RecognitionError('AUTH_REQUIRED')
  return check(
    await fetch(`${apiBase}${path}`, {
      ...options,
      cache: 'no-store',
      headers: { Authorization: `Bearer ${auth.token}` },
      signal: options.signal ?? AbortSignal.timeout(15000),
    }),
  )
}
export async function submitRecognition(file: File, roi?: number[]): Promise<RecognitionTask> {
  const body = new FormData()
  body.append('file', file)
  if (roi) body.append('roi', JSON.stringify(roi))
  return (
    await request(
      '/api/landslide/classify',
      { method: 'POST', body, signal: AbortSignal.timeout(60000) },
      true,
    )
  ).json()
}
export async function getRecognition(id: string, signal?: AbortSignal): Promise<RecognitionTask> {
  return (await request(`/api/recognition/tasks/${encodeURIComponent(id)}`, { signal })).json()
}
export async function recognitionImage(id: string, signal?: AbortSignal): Promise<Blob> {
  return (
    await request(`/api/recognition/tasks/${encodeURIComponent(id)}/image`, { signal })
  ).blob()
}
export async function recognitionHistory(
  page = 1,
): Promise<{ items: RecognitionTask[]; total: number }> {
  if (!apiBase || !savedSession()) return { items: [], total: 0 }
  return (await request(`/api/recognition/tasks?page=${page}&pageSize=20`)).json()
}
export async function deleteRecognition(id: string) {
  await request(`/api/recognition/tasks/${encodeURIComponent(id)}`, { method: 'DELETE' })
}
export function taskLabel(task: RecognitionTask) {
  return {
    queued: '排队中',
    running: '正在识别',
    completed: '识别完成 · 待复核',
    failed: '识别未完成',
  }[task.status]
}
