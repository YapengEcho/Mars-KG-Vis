import { beforeEach, describe, expect, it, vi } from 'vitest'

const memory = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (key: string) => memory.get(key) ?? null,
  setItem: (key: string, value: string) => void memory.set(key, value),
  removeItem: (key: string) => void memory.delete(key),
})

const { atlas } = await import('./atlas')

describe('saved records', () => {
  beforeEach(() => {
    memory.clear()
    atlas.state.savedIds = []
  })
  it('lists the most recently saved record first', () => {
    atlas.toggleSaved('DEMO-001')
    atlas.toggleSaved('DEMO-002')
    expect(atlas.saved.value.map((r) => r.id)).toEqual(['DEMO-002', 'DEMO-001'])
  })
  it('ignores saved ids that are absent from the current catalog', () => {
    atlas.state.savedIds = ['DEMO-001', 'GONE-1']
    expect(atlas.saved.value.map((r) => r.id)).toEqual(['DEMO-001'])
    expect(atlas.missingSaved.value).toEqual(['GONE-1'])
  })
  it('prunes stale ids and persists the remaining ones', () => {
    atlas.state.savedIds = ['DEMO-001', 'GONE-1']
    atlas.pruneSaved()
    expect(atlas.state.savedIds).toEqual(['DEMO-001'])
    expect(atlas.missingSaved.value).toEqual([])
    expect(memory.get('mars-atlas:saved:v1')).toBe(JSON.stringify(['DEMO-001']))
  })
})
