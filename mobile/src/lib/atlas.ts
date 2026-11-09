import type { Slide, SlideType } from '../types'
import { slideTypes } from '../types'
import { findSlide } from './catalog'
import rawContent from '../content/atlas.json'

export interface AtlasCitation {
  documentId: string
  locator: string
}
export interface AtlasTypeEntry {
  type: SlideType
  name: string
  english: string
  quote: string
  deposits: string
  flow: string
  recognize: string[]
  confusable: string
  citations: AtlasCitation[]
  caseIds: string[]
}
export interface AtlasComparisonRow {
  dimension: string
  left: string
  right: string
}
export interface AtlasComparison {
  id: string
  title: string
  why: string
  left: SlideType
  right: SlideType
  rows: AtlasComparisonRow[]
  citations: AtlasCitation[]
}
export interface AtlasCase {
  id: string
  landslideId: string
  type: SlideType
  title: string
  subtitle: string
  region: string
  tile: string
  previewScale: string
  observations: string[]
  tags: string[]
  citationLocator: string
}
export interface AtlasContent {
  version: string
  sourceDocument: { id: string; title: string; url: string; note: string }
  dataNote: string
  types: AtlasTypeEntry[]
  comparisons: AtlasComparison[]
  limits: string[]
  cases: AtlasCase[]
}

const casePattern = /^ls_\d{4}$/
const caseIdPattern = /^case-ls-\d{4}$/

function text(value: unknown, what: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`图鉴内容缺少${what}`)
  return value
}
function texts(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || !value.length || value.some((v) => typeof v !== 'string' || !v.trim()))
    throw new Error(`图鉴内容缺少${what}`)
  return value as string[]
}
function citations(value: unknown, what: string): AtlasCitation[] {
  if (!Array.isArray(value) || !value.length) throw new Error(`图鉴内容缺少${what}`)
  return (value as AtlasCitation[]).map((c) => {
    if (!c || typeof c.locator !== 'string' || !c.locator.trim())
      throw new Error(`图鉴内容的${what}缺少引用位置`)
    return { documentId: text(c.documentId, '引用文档编号'), locator: c.locator }
  })
}
function slideType(value: unknown, what: string): SlideType {
  if (!slideTypes.includes(value as SlideType)) throw new Error(`${what}不是有效的滑坡类别`)
  return value as SlideType
}

/**
 * Validates the hand-curated atlas content. The file carries text, citations
 * and record ids only — coordinates, areas and images are joined at runtime, so
 * the public repository never holds a copy of the private catalogue.
 */
export function parseAtlasContent(value: unknown): AtlasContent {
  if (!value || typeof value !== 'object') throw new Error('图鉴内容格式无效')
  const c = value as Partial<AtlasContent>
  const version = text(c.version, '版本号')
  const source = c.sourceDocument
  if (!source || typeof source !== 'object') throw new Error('图鉴内容缺少来源文献')
  const sourceDocument = {
    id: text(source.id, '来源文献编号'),
    title: text(source.title, '来源文献标题'),
    url: text(source.url, '来源文献地址'),
    note: typeof source.note === 'string' ? source.note : '',
  }
  if (!sourceDocument.url.startsWith('https://')) throw new Error('来源文献地址必须为 HTTPS')
  if (!Array.isArray(c.types) || c.types.length !== slideTypes.length)
    throw new Error('图鉴必须覆盖四个滑坡类别')
  const typeSet = new Set<SlideType>()
  for (const entry of c.types as AtlasTypeEntry[]) {
    const type = slideType(entry?.type, '图鉴类型')
    if (typeSet.has(type)) throw new Error('图鉴类型重复')
    typeSet.add(type)
    text(entry.name, '类型名称')
    text(entry.english, '类型英文名')
    text(entry.quote, '类型定义')
    text(entry.deposits, '堆积特征')
    text(entry.flow, '流动特征')
    text(entry.confusable, '易混类型说明')
    texts(entry.recognize, '识别要点')
    citations(entry.citations, '类型引用')
    texts(entry.caseIds, '类型案例')
  }
  if (!Array.isArray(c.cases) || !c.cases.length) throw new Error('图鉴缺少案例')
  const cases = (c.cases as AtlasCase[]).map((item) => {
    const id = text(item?.id, '案例编号')
    if (!caseIdPattern.test(id)) throw new Error(`案例编号格式无效：${id}`)
    const landslideId = text(item.landslideId, '案例对应的目录编号')
    if (!casePattern.test(landslideId)) throw new Error(`案例的目录编号格式无效：${landslideId}`)
    const entry: AtlasCase = {
      id,
      landslideId,
      type: slideType(item.type, `案例 ${id} 的类别`),
      title: text(item.title, '案例标题'),
      subtitle: text(item.subtitle, '案例副标题'),
      region: text(item.region, '案例区域'),
      tile: text(item.tile, '案例图幅'),
      previewScale: text(item.previewScale, '案例影像尺度'),
      observations: texts(item.observations, '案例观察要点'),
      tags: texts(item.tags, '案例标签'),
      citationLocator: text(item.citationLocator, '案例引用位置'),
    }
    return entry
  })
  const caseIds = new Set<string>()
  for (const item of cases) {
    if (caseIds.has(item.id)) throw new Error(`案例编号重复：${item.id}`)
    caseIds.add(item.id)
    const entry = (c.types as AtlasTypeEntry[]).find((t) => t.type === item.type)
    if (!entry?.caseIds.includes(item.id))
      throw new Error(`案例 ${item.id} 未登记在 ${item.type} 的案例清单里`)
  }
  if (!Array.isArray(c.comparisons) || c.comparisons.length < 2)
    throw new Error('图鉴至少需要两组类型对比')
  const comparisons = (c.comparisons as AtlasComparison[]).map((item) => {
    const left = slideType(item?.left, '对比左侧类型')
    const right = slideType(item.right, '对比右侧类型')
    if (left === right) throw new Error('对比两侧不能是同一类别')
    if (!Array.isArray(item.rows) || !item.rows.length) throw new Error('对比缺少对比项')
    return {
      id: text(item.id, '对比编号'),
      title: text(item.title, '对比标题'),
      why: text(item.why, '对比说明'),
      left,
      right,
      rows: item.rows.map((row) => ({
        dimension: text(row?.dimension, '对比维度'),
        left: text(row.left, '对比左侧描述'),
        right: text(row.right, '对比右侧描述'),
      })),
      citations: citations(item.citations, '对比引用'),
    }
  })
  return {
    version,
    sourceDocument,
    dataNote: typeof c.dataNote === 'string' ? c.dataNote : '',
    types: c.types as AtlasTypeEntry[],
    comparisons,
    limits: texts(c.limits, '判读限制'),
    cases,
  }
}

export const atlasContent: AtlasContent = parseAtlasContent(rawContent)

export function typeEntry(type: SlideType): AtlasTypeEntry | undefined {
  return atlasContent.types.find((entry) => entry.type === type)
}
export function casesOfType(type: SlideType): AtlasCase[] {
  return atlasContent.cases.filter((item) => item.type === type)
}
export function caseById(id: string): AtlasCase | undefined {
  return atlasContent.cases.find((item) => item.id === id)
}
/** Case page join: the catalogue record is looked up by its landslide id. */
export function caseRecord(item: AtlasCase, records: Slide[]): Slide | undefined {
  return findSlide(records, item.landslideId)
}
/** Plain-text blocks handed to the Agent so answers quote the same wording. */
export function atlasSearchText(content: AtlasContent = atlasContent): string[] {
  const blocks: string[] = []
  for (const entry of content.types) {
    blocks.push(
      [
        `类型 ${entry.type} ${entry.name}（${entry.english}）`,
        `定义：${entry.quote}`,
        `堆积特征：${entry.deposits}`,
        `流动特征：${entry.flow}`,
        `识别要点：${entry.recognize.join('；')}`,
        `易混类型：${entry.confusable}`,
        `引用：${entry.citations.map((c) => c.locator).join('；')}`,
      ].join('\n'),
    )
  }
  for (const item of content.cases) {
    const entry = typeEntry(item.type)
    blocks.push(
      [
        `案例 ${item.id}（${item.type} ${entry?.name ?? item.type}）${item.title}`,
        `副标题：${item.subtitle}`,
        `${item.region} · ${item.tile} · ${item.previewScale}`,
        `观察要点：${item.observations.join('；')}`,
        `标签：${item.tags.join('；')}`,
        `引用：${item.citationLocator}`,
      ].join('\n'),
    )
  }
  for (const item of content.comparisons) {
    blocks.push(
      [
        `类型对比 ${item.title}`,
        item.why,
        ...item.rows.map((row) => `${row.dimension}：${item.left} ${row.left} / ${item.right} ${row.right}`),
      ].join('\n'),
    )
  }
  return blocks
}
