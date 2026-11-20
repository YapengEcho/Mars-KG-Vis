import data from '../../../examples/recognition.samples.json'
import type { Slide, SlideType } from '../types'
import { publicAssetUrl } from './publicAssets'
export const samples = data.records.map((r) => ({ ...r, type: r.type as SlideType }))
// Recheck these examples when either stage of the recognition model changes.
const trialSampleIds = new Set(['ls_0493', 'ls_0483', 'ls_0597', 'ls_0482'])
export const trialSamples = samples.filter((sample) => trialSampleIds.has(sample.id))
export function sampleById(id: string) {
  return samples.find((s) => s.id === id)
}

export function caseImageUrl(record: Slide | undefined, base: string): string | undefined {
  if (!record) return undefined
  const sample = sampleById(record.id)
  if (sample) return publicAssetUrl(sample.imageUrl)
  if (base && record.imageUrl?.startsWith('/case-images/')) return base + record.imageUrl
}
