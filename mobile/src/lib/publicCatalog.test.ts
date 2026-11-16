import { describe, expect, it } from 'vitest'
import raw from '../../../examples/catalog.real-subset.json'
import { publicCatalog } from './publicCatalog'
import { parseCatalog } from './catalog'
import { coordinateStatus } from './coordinateStatus'
describe('public real-record view', () => {
  it('uses the eight approved records and keeps their original coordinates unchanged', () => {
    expect(publicCatalog.mode).toBe('reference')
    expect(publicCatalog.records).toHaveLength(8)
    expect(
      publicCatalog.records.every((r) => !r.isDemo && r.imageUrl?.startsWith('/samples/')),
    ).toBe(true)
    expect(publicCatalog.records.map((r) => [r.sourceId, r.longitude, r.latitude])).toEqual(
      raw.records.map((r) => [r.sourceId, r.longitude, r.latitude]),
    )
  })
  it('rejects promoting unverified latitude to an official catalog', () => {
    expect(() => parseCatalog({ ...publicCatalog, mode: 'catalog' })).toThrow()
    expect(() =>
      parseCatalog({
        ...publicCatalog,
        coordinateSystem: {
          ...publicCatalog.coordinateSystem,
          latitude: 'planetocentric, degrees',
        },
      }),
    ).toThrow()
  })
  it('does not extend the image-alignment result to changed or unrelated coordinates', () => {
    expect(coordinateStatus(publicCatalog).label).toBe('已与 CTX 影像对齐')
    const changed = structuredClone(publicCatalog)
    changed.records[0]!.latitude += 0.1
    expect(coordinateStatus(changed).label).toBe('原始编目坐标')
    const unrelated = structuredClone(publicCatalog)
    unrelated.records[0]!.sourceId = 999999
    expect(coordinateStatus(unrelated).note).toContain('尚未完成影像对齐核验')
    expect(coordinateStatus({ ...publicCatalog, records: [] }).label).toBe('原始编目坐标')
  })
})
