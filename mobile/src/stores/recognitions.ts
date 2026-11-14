import { reactive } from 'vue'
import { parseClassification, type ClassificationResult, type RecognitionRecord } from '../lib/recognition'

const storageKey = 'mars-atlas:recognitions:v1'
const maxRecords = 20

function normalize(value: unknown): RecognitionRecord | null {
  const parsed = parseClassification(value)
  if (!parsed) return null
  const body = value as Record<string, unknown>
  if (typeof body.id !== 'string' || typeof body.createdAt !== 'string') return null
  return {
    ...parsed,
    id: body.id,
    createdAt: body.createdAt,
    fileName: typeof body.fileName === 'string' ? body.fileName : '',
    image: typeof body.image === 'string' ? body.image : '',
  }
}

function readRecords(): RecognitionRecord[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(storageKey) ?? '[]')
    if (!Array.isArray(value)) return []
    return value.map(normalize).filter((r): r is RecognitionRecord => Boolean(r))
  } catch {
    return []
  }
}

function write(): boolean {
  try {
    localStorage.setItem(storageKey, JSON.stringify(state.records))
    state.storageError = ''
    return true
  } catch {
    // 缩略图最占空间：丢弃较早记录的图片后重试，文字结果尽量保留。
    let stripped = false
    state.records = state.records.map((record, index) => {
      if (index < 3 || !record.image) return record
      stripped = true
      return { ...record, image: '' }
    })
    try {
      localStorage.setItem(storageKey, JSON.stringify(state.records))
      state.storageError = stripped ? '本机存储空间不足，较早记录仅保留文字结果。' : ''
      return true
    } catch {
      state.storageError = '当前设备无法保存识别记录，请检查存储权限。'
      return false
    }
  }
}

const state = reactive({
  records: readRecords(),
  storageError: '',
})

export const recognitions = {
  state,
  get(id: string) {
    return state.records.find((record) => record.id === id)
  },
  add(result: ClassificationResult, fileName: string, image: string): RecognitionRecord {
    const record: RecognitionRecord = {
      ...result,
      id: `rec-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      createdAt: new Date().toISOString(),
      fileName,
      image,
    }
    state.records = [record, ...state.records].slice(0, maxRecords)
    write()
    return record
  },
  remove(id: string): void {
    state.records = state.records.filter((record) => record.id !== id)
    write()
  },
}
