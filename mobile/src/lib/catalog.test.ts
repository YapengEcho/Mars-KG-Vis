import { describe, expect, it } from 'vitest'
import { filterSlides, findSlide, parseCatalog } from './catalog'
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
  it('accepts a fully-formed catalog-mode record set', () => {
    const catalog = {
      version: 'catalog-1',
      mode: 'catalog',
      coordinateSystem: {
        body: 'Mars',
        longitude: 'east-positive, -180..180',
        latitude: 'planetocentric, degrees',
      },
      records: [
        {
          id: 'REAL-001',
          sourceId: 1001,
          name: '真实点位 01',
          type: 'SL',
          longitude: -68.2,
          latitude: -12.4,
          areaKm2: 210.5,
          summary: '真实目录样例，用于验证非示例数据链路。',
          imageUrl: null,
          source: { title: '真实来源', url: null },
          isDemo: false,
        },
      ],
    }
    const parsed = parseCatalog(catalog)
    expect(parsed.mode).toBe('catalog')
    expect(parsed.records[0]?.isDemo).toBe(false)
    expect(parsed.records[0]?.sourceId).toBe(1001)
  })
  it('combines case-insensitive search with the selected type', () => {
    const catalog = parseCatalog(demo)
    expect(filterSlides(catalog.records, 'demo-001', 'SL').map((x) => x.id)).toEqual(['DEMO-001'])
    expect(filterSlides(catalog.records, 'demo-001', 'FL')).toEqual([])
  })
  it('searches the same fields as the API list endpoint', () => {
    const catalog = parseCatalog(demo)
    expect(filterSlides(catalog.records, '程序构造', '')).toHaveLength(20)
    expect(filterSlides(catalog.records, '002', 'SP').map((x) => x.id)).toEqual(['DEMO-002'])
    expect(filterSlides(catalog.records, '002', 'FL')).toEqual([])
  })
  it('resolves focus links by exact record id or original number', () => {
    const records = parseCatalog(demo).records
    expect(findSlide(records, 'demo-001')?.id).toBe('DEMO-001')
    const real = { ...records[0]!, id: 'REAL-001', sourceId: 1001, isDemo: false }
    expect(findSlide([real], '1001')?.id).toBe('REAL-001')
    expect(findSlide([real], 'REAL-001')?.id).toBe('REAL-001')
    expect(findSlide([real], '100')).toBeUndefined()
    expect(findSlide([real], ' ')).toBeUndefined()
  })
})
