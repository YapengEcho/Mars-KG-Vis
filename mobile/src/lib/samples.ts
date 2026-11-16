import data from '../../../examples/recognition.samples.json'
import type { SlideType } from '../types'
export const samples = data.records.map((r) => ({ ...r, type: r.type as SlideType }))
export function sampleById(id: string) {
  return samples.find((s) => s.id === id)
}
