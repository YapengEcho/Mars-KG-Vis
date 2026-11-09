import { describe, expect, it } from 'vitest'
import {
  atlasContent,
  atlasSearchText,
  caseById,
  caseRecord,
  casesOfType,
  parseAtlasContent,
  typeEntry,
} from './atlas'
import raw from '../content/atlas.json'
import demo from '../../../examples/catalog.demo.json'
import type { Slide } from '../types'

const clone = () => structuredClone(raw) as unknown as Record<string, unknown>

describe('atlas content', () => {
  it('covers the four catalogue types with three cases each', () => {
    expect(atlasContent.types.map((t) => t.type)).toEqual(['SL', 'SP', 'FL', 'CL'])
    for (const type of ['SL', 'SP', 'FL', 'CL'] as const) {
      expect(casesOfType(type)).toHaveLength(3)
    }
    expect(atlasContent.cases).toHaveLength(12)
    expect(atlasContent.comparisons).toHaveLength(2)
  })
  it('quotes the source paper for every type', () => {
    for (const entry of atlasContent.types) {
      expect(entry.quote.length).toBeGreaterThan(20)
      expect(entry.citations.some((c) => c.locator.includes('表 2'))).toBe(true)
      expect(entry.citations.every((c) => c.documentId === 'doc-inventory-paper')).toBe(true)
    }
  })
  it('keeps catalogue values out of the content file', () => {
    for (const item of atlasContent.cases) {
      expect(item).not.toHaveProperty('longitude')
      expect(item).not.toHaveProperty('latitude')
      expect(item).not.toHaveProperty('areaKm2')
      expect(item).not.toHaveProperty('sourceId')
      expect(item.landslideId).toMatch(/^ls_\d{4}$/)
    }
  })
  it('rejects a case that is not registered under its type', () => {
    const content = clone()
    const types = content.types as { type: string; caseIds: string[] }[]
    types[0]!.caseIds = types[0]!.caseIds.filter((id) => id !== 'case-ls-0493')
    expect(() => parseAtlasContent(content)).toThrow(/未登记/)
  })
  it('rejects duplicate case identifiers', () => {
    const content = clone()
    const cases = content.cases as { id: string }[]
    cases[1]!.id = cases[0]!.id
    expect(() => parseAtlasContent(content)).toThrow(/重复/)
  })
  it('rejects comparisons between the same type', () => {
    const content = clone()
    const comparisons = content.comparisons as { right: string }[]
    comparisons[0]!.right = 'SL'
    expect(() => parseAtlasContent(content)).toThrow()
  })
  it('rejects a type without recognition points', () => {
    const content = clone()
    ;(content.types as { recognize: string[] }[])[0]!.recognize = []
    expect(() => parseAtlasContent(content)).toThrow(/识别要点/)
  })
})

describe('case to catalogue join', () => {
  it('resolves nothing against the demo catalogue', () => {
    const item = caseById('case-ls-0493')!
    expect(caseRecord(item, demo.records as unknown as Slide[])).toBeUndefined()
  })
  it('resolves a real record by its landslide id', () => {
    const item = caseById('case-ls-0493')!
    const record: Slide = {
      id: 'ls_0493',
      sourceId: 493,
      name: '滑坡 493',
      type: 'SL',
      longitude: -82.1148,
      latitude: -4.6952,
      areaKm2: 200.48,
      summary: '',
      imageUrl: null,
      source: { title: '真实来源', url: null },
      isDemo: false,
    }
    expect(caseRecord(item, [record])?.id).toBe('ls_0493')
    expect(caseRecord(item, [{ ...record, id: 'ls_0494' }])).toBeUndefined()
  })
  it('finds type entries and cases by id', () => {
    expect(typeEntry('SP')?.name).toBe('滑塌型')
    expect(caseById('case-ls-0592')?.type).toBe('CL')
    expect(caseById('case-ls-0000')).toBeUndefined()
  })
})

describe('searchable text for the agent', () => {
  const blocks = atlasSearchText()
  it('contains every type and case once', () => {
    expect(blocks).toHaveLength(4 + 12 + 2)
    for (const type of ['类型 SL', '类型 SP', '类型 FL', '类型 CL']) {
      expect(blocks.filter((b) => b.startsWith(type))).toHaveLength(1)
    }
    for (const item of atlasContent.cases) expect(blocks).toContainEqual(expect.stringContaining(item.id))
  })
  it('carries the citation locators and no coordinates', () => {
    expect(blocks.join('\n')).toContain('表 2')
    expect(blocks.join('\n')).not.toMatch(/°W|°E/)
  })
})
