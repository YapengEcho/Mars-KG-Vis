import { describe, expect, it } from 'vitest'
import { parseRing, parseSlideGeometry } from './geometry'

const polygon = {
  status: 'ok',
  feature: {
    type: 'Feature',
    properties: { kind: 'polygon' },
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [-68.5, -12.7],
          [-67.9, -12.7],
          [-67.9, -12.1],
          [-68.5, -12.1],
          [-68.5, -12.7],
        ],
      ],
    },
  },
  canOverlay: true,
  coordinateSystem: { latitude: 'planetocentric' },
}

describe('boundary geometry', () => {
  it('keeps only well-formed rings', () => {
    expect(parseRing([[-68.5, -12.7], [-67.9, -12.7], [-67.9, -12.1]])).toHaveLength(3)
    expect(parseRing([[-68.5, -12.7], [-67.9, -12.7]])).toBeNull()
    expect(parseRing([[10, 91], [11, 20], [11, 21]])).toBeNull()
    expect(parseRing([[10, 20], [11, 20], ['a', 21]])).toBeNull()
  })
  it('reads the outer ring of a GeoJSON polygon response', () => {
    const parsed = parseSlideGeometry(polygon)
    expect(parsed?.ring).toHaveLength(5)
    expect(parsed?.canOverlay).toBe(true)
    expect(parsed?.latitudeConvention).toBe('planetocentric')
  })
  it('reads the first ring of a multipolygon and of a bare feature', () => {
    const multi = {
      feature: {
        type: 'Feature',
        geometry: {
          type: 'MultiPolygon',
          coordinates: [polygon.feature.geometry.coordinates],
        },
      },
      canOverlay: true,
    }
    expect(parseSlideGeometry(multi)?.ring).toHaveLength(5)
    expect(parseSlideGeometry(polygon.feature)?.ring).toHaveLength(5)
  })
  it('keeps the overlay flag when the coordinate convention is unconfirmed', () => {
    const parsed = parseSlideGeometry({
      ...polygon,
      canOverlay: false,
      coordinateSystem: { latitude: 'planetographic' },
    })
    expect(parsed?.canOverlay).toBe(false)
    expect(parsed?.latitudeConvention).toBe('planetographic')
  })
  it('rejects incomplete or unrelated payloads', () => {
    expect(parseSlideGeometry(null)).toBeNull()
    expect(parseSlideGeometry({ feature: null })).toBeNull()
    expect(parseSlideGeometry({ feature: { geometry: { type: 'Point', coordinates: [1, 2] } } })).toBeNull()
    expect(
      parseSlideGeometry({ feature: { geometry: { type: 'Polygon', coordinates: [] } } }),
    ).toBeNull()
  })
})
