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
export interface RecognitionScreening {
  decision: 'possible_landslide' | 'no_landslide' | 'unsupported_image' | 'inconclusive'
  score: number | null
  threshold: number | null
  modelVersion: string | null
}
export interface RecognitionTask {
  taskId: string
  kind: 'classify'
  status: 'queued' | 'running' | 'completed' | 'failed'
  createdAt: string
  finishedAt: string | null
  modelVersion: string
  preprocessingVersion: string
  result: RecognitionResult | null
  screening?: RecognitionScreening | null
  errorCode: string | null
}
export interface DetectionCandidate {
  candidateId: string
  box: [number, number, number, number]
  detectionScore: number
  classification: RecognitionResult
}
export interface DetectionResult {
  outcome: 'detections' | 'no_detections'
  coordinateSpace: 'image_pixels'
  truncated: boolean
  candidates: DetectionCandidate[]
}
export interface DetectionTask {
  taskId: string
  kind: 'detect'
  status: 'queued' | 'running' | 'completed' | 'failed'
  createdAt: string
  finishedAt: string | null
  modelVersion: string
  preprocessingVersion: string
  input: { width: number; height: number; preprocessingVersion: string }
  result: DetectionResult | null
  errorCode: string | null
  imageUrl: string
}
export type HistoryTask = RecognitionTask | DetectionTask
export type HistoryKind = 'classify' | 'detect'
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
  DETECTION_NOT_CONFIGURED: '检测服务尚未就绪，请稍后重试。',
  SCREENING_NOT_CONFIGURED: '滑坡初筛服务尚未就绪，请稍后重试。',
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
/**
 * Device credential for a chat request that explains a selected task.
 *
 * Reuses an existing session only:聊天本身不应该新建设备身份，否则会在用户
 * 没有上传图片时凭空产生一条会话记录。
 */
export async function recognitionAuthHeaders(): Promise<Record<string, string> | null> {
  const auth = await session(false)
  return auth ? { Authorization: `Bearer ${auth.token}` } : null
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
export async function submitDetection(file: File): Promise<DetectionTask> {
  const body = new FormData()
  body.append('file', file)
  return (
    await request(
      '/api/landslide/detect',
      { method: 'POST', body, signal: AbortSignal.timeout(60000) },
      true,
    )
  ).json()
}
export async function getRecognition(id: string, signal?: AbortSignal): Promise<RecognitionTask> {
  return (await request(`/api/recognition/tasks/${encodeURIComponent(id)}`, { signal })).json()
}
export async function getDetection(id: string, signal?: AbortSignal): Promise<DetectionTask> {
  return (await request(`/api/recognition/tasks/${encodeURIComponent(id)}`, { signal })).json()
}
export async function recognitionImage(id: string, signal?: AbortSignal): Promise<Blob> {
  return (
    await request(`/api/recognition/tasks/${encodeURIComponent(id)}/image`, { signal })
  ).blob()
}
export async function recognitionHistory(
  page = 1,
  kind: HistoryKind = 'classify',
): Promise<{ items: HistoryTask[]; total: number }> {
  if (!apiBase || !savedSession()) return { items: [], total: 0 }
  return (
    await request(`/api/recognition/tasks?kind=${kind}&page=${page}&pageSize=20`)
  ).json()
}
export async function deleteRecognition(id: string) {
  await request(`/api/recognition/tasks/${encodeURIComponent(id)}`, { method: 'DELETE' })
}
export function taskLabel(task: HistoryTask) {
  if (task.kind === 'detect') {
    if (task.status === 'completed') {
      if (!task.result) return '检测完成'
      return task.result.outcome === 'no_detections'
        ? '未检出候选 · 待复核'
        : `检出 ${task.result.candidates.length} 个候选 · 待复核`
    }
    return { queued: '排队中', running: '正在检测', failed: '检测未完成' }[task.status]
  }
  if (task.status === 'completed') {
    if (task.screening) {
      return {
        possible_landslide: '疑似滑坡 · 待复核',
        no_landslide: '未检出明显滑坡',
        unsupported_image: '影像不适用',
        inconclusive: '初筛结果不确定',
      }[task.screening.decision]
    }
    return task.result ? '旧记录 · 未初筛' : '识别完成'
  }
  return {
    queued: '排队中',
    running: '正在识别',
    failed: '识别未完成',
  }[task.status]
}
