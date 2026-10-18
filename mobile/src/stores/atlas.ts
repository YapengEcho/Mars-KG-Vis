import { computed, reactive } from 'vue'
import demo from '../../../examples/catalog.demo.json'
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
  catalog: parseCatalog(demo) as Catalog,
  selectedId: '' as string,
  savedIds: readSaved(),
  error: '',
  storageError: '',
  loading: false,
})
export const atlas = {
  state,
  selected: computed(() => state.catalog.records.find((r) => r.id === state.selectedId)),
  saved: computed(() => state.catalog.records.filter((r) => state.savedIds.includes(r.id))),
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
    const url =
      import.meta.env.VITE_CATALOG_URL?.trim() || (apiBase ? `${apiBase}/api/catalog` : '')
    if (!url) return
    state.loading = true
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(12000) })
      if (!r.ok) throw new Error()
      state.catalog = parseCatalog(await r.json())
      state.error = ''
    } catch {
      state.error = '目录加载失败，当前显示示例目录。'
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
