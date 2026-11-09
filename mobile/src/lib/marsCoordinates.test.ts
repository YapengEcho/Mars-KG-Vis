import { describe, expect, it } from 'vitest'
import {
  MARS_RADIUS_KM,
  angularDistance,
  cartesianToLonLat,
  isMarsCoordinate,
  lonLatToCartesian,
  normalizeLongitude,
  surfaceDistanceKm,
} from './marsCoordinates'

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
  it('rejects coordinates outside the Mars ranges', () => {
    expect(isMarsCoordinate({ longitude: -68.2, latitude: -12.4 })).toBe(true)
    expect(isMarsCoordinate({ longitude: 180, latitude: 90 })).toBe(true)
    expect(isMarsCoordinate({ longitude: 999, latitude: 0 })).toBe(false)
    expect(isMarsCoordinate({ longitude: 0, latitude: 91 })).toBe(false)
    expect(isMarsCoordinate({ longitude: Number.NaN, latitude: 0 })).toBe(false)
  })
  it('measures great-circle distance on the Mars sphere', () => {
    expect(angularDistance(-68.2, -12.4, -68.2, -12.4)).toBeCloseTo(0, 10)
    expect(angularDistance(0, 0, 180, 0)).toBeCloseTo(180, 10)
    expect(surfaceDistanceKm(0, 0, 180, 0)).toBeCloseTo(Math.PI * MARS_RADIUS_KM, 6)
    // One degree of latitude at the equator equals one degree of arc.
    expect(surfaceDistanceKm(0, 0, 0, 1)).toBeCloseTo((Math.PI / 180) * MARS_RADIUS_KM, 6)
  })
})
