import type { Catalog, Slide, SlideType } from '../types'
import { slideTypes } from '../types'

export function parseCatalog(value: unknown): Catalog {
  if (!value || typeof value !== 'object') throw new Error('目录格式无效')
  const c = value as Partial<Catalog>
  if (
    !Array.isArray(c.records) ||
    !c.records.length ||
    !['demo', 'catalog'].includes(c.mode ?? '') ||
    typeof c.version !== 'string'
  )
    throw new Error('目录缺少版本、数据模式或记录')
  if (
    c.coordinateSystem?.body !== 'Mars' ||
    c.coordinateSystem.longitude !== 'east-positive, -180..180' ||
    c.coordinateSystem.latitude !== 'planetocentric, degrees'
  )
    throw new Error('目录必须使用约定的火星坐标')
  const ids = new Set<string>()
  for (const r of c.records) {
    if (
      !r ||
      typeof r.id !== 'string' ||
      !r.id ||
      ids.has(r.id) ||
      !slideTypes.includes(r.type) ||
      typeof r.name !== 'string' ||
      typeof r.summary !== 'string'
    )
      throw new Error('目录编号、名称或类别无效')
    if (
      !Number.isFinite(r.longitude) ||
      Math.abs(r.longitude) > 180 ||
      !Number.isFinite(r.latitude) ||
      Math.abs(r.latitude) > 90
    )
      throw new Error('目录坐标超出范围')
    if (
      typeof r.isDemo !== 'boolean' ||
      r.isDemo !== (c.mode === 'demo') ||
      !r.source ||
      typeof r.source.title !== 'string'
    )
      throw new Error('目录来源或示例状态无效')
    if (r.sourceId !== null && (!Number.isInteger(r.sourceId) || r.sourceId < 0))
      throw new Error('原始编号无效')
    if (
      r.source.url !== null &&
      (typeof r.source.url !== 'string' || !r.source.url.startsWith('https://'))
    )
      throw new Error('来源链接必须为HTTPS或空值')
    if (r.areaKm2 !== null && (!Number.isFinite(r.areaKm2) || r.areaKm2 < 0))
      throw new Error('面积必须为非负数或空值')
    if (
      r.imageUrl !== null &&
      (typeof r.imageUrl !== 'string' ||
        r.imageUrl.startsWith('//') ||
        (!r.imageUrl.startsWith('/') && !r.imageUrl.startsWith('https://')))
    )
      throw new Error('影像地址无效')
    ids.add(r.id)
  }
  return c as Catalog
}
/**
 * Mirrors the API's `q` matching: id, source number, name and summary, so the
 * APP and the Agent return the same records for the same query.
 */
export function filterSlides(records: Slide[], query: string, type: SlideType | ''): Slide[] {
  const key = query.trim().toLocaleLowerCase()
  return records.filter(
    (r) =>
      (!type || r.type === type) &&
      (!key ||
        `${r.id} ${r.sourceId ?? ''} ${r.name} ${r.summary}`.toLocaleLowerCase().includes(key)),
  )
}

/** Resolves an exact record id or original number used by `?focus=` links. */
export function findSlide(records: Slide[], key: string): Slide | undefined {
  const term = key.trim().toLocaleLowerCase()
  if (!term) return undefined
  return records.find(
    (r) => r.id.toLocaleLowerCase() === term || String(r.sourceId ?? '') === term,
  )
}
