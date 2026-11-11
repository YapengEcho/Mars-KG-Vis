import { computed, reactive } from 'vue'
import { publicCatalog } from '../lib/publicCatalog'
import type { Catalog, Slide } from '../types'
import { parseCatalog } from '../lib/catalog'
import { apiBase } from '../services/api'

const storageKey = 'mars-atlas:saved:v1'
function readSaved(): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(storageKey) ?? '[]')
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === 'string') : []
  } catch {
    return []
  }
}
const state = reactive({
  catalog: publicCatalog as Catalog,
  selectedId: '' as string,
  savedIds: readSaved(),
  error: '',
  storageError: '',
  loading: false,
})
export const atlas = {
  state,
  selected: computed(() => state.catalog.records.find((r) => r.id === state.selectedId)),
  saved: computed(() =>
    [...state.savedIds]
      .reverse()
      .map((id) => state.catalog.records.find((r) => r.id === id))
      .filter((r): r is Slide => Boolean(r)),
  ),
  /** Collected ids that no longer exist in the currently loaded catalog. */
  missingSaved: computed(() =>
    state.savedIds.filter((id) => !state.catalog.records.some((r) => r.id === id)),
  ),
  pruneSaved() {
    const next = state.savedIds.filter((id) => state.catalog.records.some((r) => r.id === id))
    try {
      localStorage.setItem(storageKey, JSON.stringify(next))
      state.savedIds = next
      state.storageError = ''
    } catch {
      state.storageError = '当前设备无法保存收藏，请检查存储权限。'
    }
  },
  select(id: string) {
    state.selectedId = id
  },
  dismiss() {
    state.selectedId = ''
  },
  isSaved(id: string) {
    return state.savedIds.includes(id)
  },
  toggleSaved(id: string) {
    const next = state.savedIds.includes(id)
      ? state.savedIds.filter((x) => x !== id)
      : [...state.savedIds, id]
    try {
      localStorage.setItem(storageKey, JSON.stringify(next))
      state.savedIds = next
      state.storageError = ''
    } catch {
      state.storageError = '当前设备无法保存收藏，请检查存储权限。'
    }
  },
  async load() {
    if (state.loading) return
    const url =
      import.meta.env.VITE_CATALOG_URL?.trim() || (apiBase ? `${apiBase}/api/catalog` : '')
    if (!url) return
    state.loading = true
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(12000) })
      if (!r.ok) throw new Error()
      const loaded = parseCatalog(await r.json())
      // An older default server must not silently replace the real records with fixtures.
      if (loaded.mode !== 'demo' || import.meta.env.VITE_CATALOG_URL?.trim()) state.catalog = loaded
      state.error = ''
    } catch {
      state.error = '目录更新失败，已保留当前记录。'
    } finally {
      state.loading = false
    }
  },
}
export function byType(records: Slide[]) {
  return records.reduce<Record<string, number>>((counts, r) => {
    counts[r.type] = (counts[r.type] ?? 0) + 1
    return counts
  }, {})
}
