import { describe, expect, it } from 'vitest'
import { fitCandidateBox } from './detection'

const image = { width: 1800, height: 1500 }

describe('fitCandidateBox', () => {
  it('centres a small candidate and zooms in', () => {
    const box: [number, number, number, number] = [600, 400, 900, 600]
    const result = fitCandidateBox(box, image, 600)
    expect(result.scale).toBeGreaterThan(1)
    // candidate centre (750, 500) → image centre (900, 750)
    const stageHeight = 600 * (1500 / 1800)
    expect(result.x).toBeCloseTo(600 / 2 - (750 / 1800) * 600 * result.scale)
    expect(result.y).toBeCloseTo(stageHeight / 2 - (500 / 1500) * stageHeight * result.scale)
  })

  it('keeps the whole image when the box covers it', () => {
    expect(fitCandidateBox([0, 0, 1800, 1500], image, 600)).toEqual({ scale: 1, x: 0, y: 0 })
  })

  it('falls back to the full view for degenerate input', () => {
    expect(fitCandidateBox([10, 10, 10, 10], image, 600)).toEqual({ scale: 1, x: 0, y: 0 })
    expect(fitCandidateBox([0, 0, 10, 10], image, 0)).toEqual({ scale: 1, x: 0, y: 0 })
  })

  it('caps extreme zoom', () => {
    expect(fitCandidateBox([100, 100, 102, 102], image, 600).scale).toBe(8)
  })
})
