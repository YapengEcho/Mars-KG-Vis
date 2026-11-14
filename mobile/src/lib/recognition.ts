import { slideTypes, type SlideType } from '../types'

/** 单图分类结果，字段与 contracts/recognition.md 的成功响应一致。 */
export interface ClassificationResult {
  prediction: SlideType
  probabilities: Record<SlideType, number>
  confidence: number
  uncertain: boolean
  modelVersion: string
}

export interface RecognitionRecord extends ClassificationResult {
  id: string
  createdAt: string
  fileName: string
  /** 本机缩略图（data URL）；生成失败时为空字符串。 */
  image: string
}

function isSlideType(value: unknown): value is SlideType {
  return typeof value === 'string' && (slideTypes as readonly string[]).includes(value)
}

function isRatio(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

/**
 * 解析分类服务返回体。prediction 必须是四类之一，probabilities 必须含四类全量分数；
 * confidence 缺失时退回预测类别分数，modelVersion 缺失时标注为“未标注”。
 */
export function parseClassification(payload: unknown): ClassificationResult | null {
  if (!payload || typeof payload !== 'object') return null
  const body = payload as Record<string, unknown>
  if (!isSlideType(body.prediction)) return null
  if (!body.probabilities || typeof body.probabilities !== 'object') return null
  const raw = body.probabilities as Record<string, unknown>
  const probabilities = {} as Record<SlideType, number>
  for (const type of slideTypes) {
    const value = raw[type]
    if (!isRatio(value)) return null
    probabilities[type] = value
  }
  const confidence = isRatio(body.confidence) ? body.confidence : probabilities[body.prediction]
  const modelVersion =
    typeof body.modelVersion === 'string' && body.modelVersion.trim()
      ? body.modelVersion.trim()
      : '未标注'
  return {
    prediction: body.prediction,
    probabilities,
    confidence,
    uncertain: body.uncertain === true,
    modelVersion,
  }
}

/** 概率百分比，保留一位小数。 */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

/** 本机记录时间，按设备时区显示到分钟。 */
export function formatRecognizedAt(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}
