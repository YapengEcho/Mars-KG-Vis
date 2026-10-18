import { describe, expect, it } from 'vitest'
import { cartesianToLonLat, lonLatToCartesian, normalizeLongitude } from './marsCoordinates'

describe('mars coordinates', () => {
  it('normalizes the same meridian without shifting the latitude', () => {
    expect(normalizeLongitude(270)).toBe(-90)
    expect(normalizeLongitude(360)).toBe(0)
  })
  it('preserves distinct hemispheres through coordinate conversion', () => {
    for (const [longitude, latitude] of [
      [0, 0],
      [90, 20],
      [-90, -30],
      [179, 65],
    ]) {
      const point = lonLatToCartesian(longitude!, latitude!, 2)
      const restored = cartesianToLonLat(...point)
      expect(restored.longitude).toBeCloseTo(longitude!, 5)
      expect(restored.latitude).toBeCloseTo(latitude!, 5)
    }
  })
})
