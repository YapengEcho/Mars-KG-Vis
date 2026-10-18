import { describe, expect, it } from 'vitest'
import { filterSlides, parseCatalog } from './catalog'
import demo from '../../../examples/catalog.demo.json'

describe('catalog boundary', () => {
  it('loads the demo without treating it as geological truth', () => {
    const catalog = parseCatalog(demo)
    expect(catalog.records).toHaveLength(20)
    expect(catalog.records.every((r) => r.isDemo && r.sourceId === null)).toBe(true)
  })
  it('rejects repeated identifiers', () => {
    const catalog = structuredClone(demo)
    catalog.records[1]!.id = catalog.records[0]!.id
    expect(() => parseCatalog(catalog)).toThrow()
  })
  it('rejects invalid coordinates and mixed data modes', () => {
    const catalog = structuredClone(demo)
    catalog.records[0]!.latitude = 91
    expect(() => parseCatalog(catalog)).toThrow()
    catalog.records[0]!.latitude = 0
    catalog.mode = 'catalog'
    expect(() => parseCatalog(catalog)).toThrow()
  })
  it('rejects malformed source URLs before opening details', () => {
    const catalog = structuredClone(demo)
    const record = catalog.records[0] as unknown as Record<string, unknown>
    record.source = { title: 'bad', url: 42 }
    expect(() => parseCatalog(catalog)).toThrow()
  })
  it('rejects earth coordinate metadata', () => {
    const catalog = structuredClone(demo)
    catalog.coordinateSystem.body = 'Earth'
    expect(() => parseCatalog(catalog)).toThrow()
  })
  it('combines case-insensitive search with the selected type', () => {
    const catalog = parseCatalog(demo)
    expect(filterSlides(catalog.records, 'demo-001', 'SL').map((x) => x.id)).toEqual(['DEMO-001'])
    expect(filterSlides(catalog.records, 'demo-001', 'FL')).toEqual([])
  })
})
