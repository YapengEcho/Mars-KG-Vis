import { describe, expect, it } from 'vitest'
import raw from '../../../examples/catalog.real-subset.json'
import { publicCatalog } from './publicCatalog'
import { parseCatalog } from './catalog'
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
})
