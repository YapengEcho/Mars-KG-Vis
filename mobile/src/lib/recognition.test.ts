import { describe, expect, it } from 'vitest'
import { formatPercent, parseClassification } from './recognition'

const payload = {
  prediction: 'CL',
  probabilities: { SL: 0.05, SP: 0.12, FL: 0.21, CL: 0.62 },
  confidence: 0.62,
  uncertain: false,
  modelVersion: 'dinov2-base-lora-r4',
}

describe('parseClassification', () => {
  it('accepts the contract response', () => {
    expect(parseClassification(payload)).toEqual({
      prediction: 'CL',
      probabilities: { SL: 0.05, SP: 0.12, FL: 0.21, CL: 0.62 },
      confidence: 0.62,
      uncertain: false,
      modelVersion: 'dinov2-base-lora-r4',
    })
  })
  it('rejects unknown predictions and incomplete probability sets', () => {
    expect(parseClassification({ ...payload, prediction: 'XX' })).toBeNull()
    expect(
      parseClassification({ ...payload, probabilities: { SL: 0.5, SP: 0.5, FL: 0.5 } }),
    ).toBeNull()
  })
  it('rejects out-of-range ratios and malformed payloads', () => {
    expect(
      parseClassification({ ...payload, probabilities: { ...payload.probabilities, CL: 1.4 } }),
    ).toBeNull()
    expect(parseClassification(null)).toBeNull()
    expect(parseClassification('ok')).toBeNull()
  })
  it('derives missing fields from the probabilities', () => {
    const parsed = parseClassification({
      prediction: 'SP',
      probabilities: { SL: 0.1, SP: 0.7, FL: 0.1, CL: 0.1 },
    })
    expect(parsed?.confidence).toBe(0.7)
    expect(parsed?.uncertain).toBe(false)
    expect(parsed?.modelVersion).toBe('未标注')
  })
  it('keeps the manual-review flag', () => {
    expect(parseClassification({ ...payload, uncertain: true })?.uncertain).toBe(true)
  })
})

describe('formatPercent', () => {
  it('renders one decimal place', () => {
    expect(formatPercent(0.6234)).toBe('62.3%')
    expect(formatPercent(1)).toBe('100.0%')
  })
})
