import { beforeEach, describe, expect, it, vi } from 'vitest'

const memory = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (key: string) => memory.get(key) ?? null,
  setItem: (key: string, value: string) => void memory.set(key, value),
  removeItem: (key: string) => void memory.delete(key),
})

const { recognitions } = await import('./recognitions')

const result = {
  prediction: 'CL' as const,
  probabilities: { SL: 0.05, SP: 0.12, FL: 0.21, CL: 0.62 },
  confidence: 0.62,
  uncertain: false,
  modelVersion: 'dinov2-base-lora-r4',
}

describe('recognition records', () => {
  beforeEach(() => {
    memory.clear()
    recognitions.state.records = []
    recognitions.state.storageError = ''
  })
  it('keeps the newest record first and persists it', () => {
    const first = recognitions.add(result, 'a.png', 'data:image/jpeg;base64,aa')
    const second = recognitions.add(result, 'b.png', '')
    expect(recognitions.state.records.map((r) => r.fileName)).toEqual(['b.png', 'a.png'])
    expect(recognitions.get(second.id)?.fileName).toBe('b.png')
    const stored = JSON.parse(memory.get('mars-atlas:recognitions:v1') ?? '[]') as { id: string }[]
    expect(stored.map((r) => r.id)).toEqual([second.id, first.id])
  })
  it('removes a record and updates storage', () => {
    const record = recognitions.add(result, 'a.png', '')
    recognitions.remove(record.id)
    expect(recognitions.state.records).toEqual([])
    expect(memory.get('mars-atlas:recognitions:v1')).toBe('[]')
  })
  it('drops records beyond the local limit', () => {
    for (let i = 0; i < 22; i += 1) recognitions.add(result, `${i}.png`, '')
    expect(recognitions.state.records).toHaveLength(20)
    expect(recognitions.state.records[0].fileName).toBe('21.png')
  })
  it('reports a storage failure without losing the in-memory record', () => {
    const spy = vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('quota')
    })
    recognitions.add(result, 'a.png', '')
    expect(recognitions.state.records).toHaveLength(1)
    expect(recognitions.state.storageError).toContain('无法保存识别记录')
    spy.mockRestore()
  })
})
