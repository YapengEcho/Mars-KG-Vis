import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import demo from '../../../examples/catalog.demo.json'
import { publicCatalog } from '../lib/publicCatalog'

vi.mock('../services/api', () => ({ apiBase: 'https://catalog.example.test' }))

const memory = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (key: string) => memory.get(key) ?? null,
  setItem: (key: string, value: string) => void memory.set(key, value),
  removeItem: (key: string) => void memory.delete(key),
})

const { atlas } = await import('./atlas')

afterEach(() => {
  vi.unstubAllEnvs()
  atlas.state.catalog = publicCatalog
  atlas.state.error = ''
})

describe('public catalog loading', () => {
  it('retains real records if an older API still returns the demo catalog', async () => {
    vi.stubEnv('VITE_CATALOG_URL', '')
    const fetcher = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify(demo)))
    try {
      await atlas.load()
      expect(atlas.state.catalog.mode).toBe('reference')
      expect(atlas.state.catalog.records.every((r) => !r.isDemo)).toBe(true)
    } finally {
      fetcher.mockRestore()
    }
  })

  it('keeps the real catalog available after a network error', async () => {
    vi.stubEnv('VITE_CATALOG_URL', '')
    const fetcher = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'))
    try {
      await atlas.load()
      expect(atlas.state.catalog.records).toHaveLength(8)
      expect(atlas.state.catalog.mode).toBe('reference')
      expect(atlas.state.error).toBeTruthy()
      expect(atlas.state.loading).toBe(false)
    } finally {
      fetcher.mockRestore()
    }
  })
})

describe('saved records', () => {
  beforeEach(() => {
    memory.clear()
    atlas.state.savedIds = []
  })
  it('lists the most recently saved record first', () => {
    atlas.toggleSaved('ls_0493')
    atlas.toggleSaved('ls_0529')
    expect(atlas.saved.value.map((r) => r.id)).toEqual(['ls_0529', 'ls_0493'])
  })
  it('ignores saved ids that are absent from the current catalog', () => {
    atlas.state.savedIds = ['ls_0493', 'GONE-1']
    expect(atlas.saved.value.map((r) => r.id)).toEqual(['ls_0493'])
    expect(atlas.missingSaved.value).toEqual(['GONE-1'])
  })
  it('prunes stale ids and persists the remaining ones', () => {
    atlas.state.savedIds = ['ls_0493', 'GONE-1']
    atlas.pruneSaved()
    expect(atlas.state.savedIds).toEqual(['ls_0493'])
    expect(atlas.missingSaved.value).toEqual([])
    expect(memory.get('mars-atlas:saved:v1')).toBe(JSON.stringify(['ls_0493']))
  })
})
