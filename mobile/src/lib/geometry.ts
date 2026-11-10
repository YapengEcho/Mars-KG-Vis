/** Closed boundary ring as [longitude, latitude] pairs, east-positive degrees. */
export type Ring = [number, number][]

export interface SlideGeometry {
  ring: Ring
  /** Whether the source latitude convention is confirmed, so it may be drawn on the globe. */
  canOverlay: boolean
  kind: string
  latitudeConvention: string
}

/** Returns a validated closed ring, or null when the value is absent/invalid. */
export function parseRing(value: unknown): Ring | null {
  if (!Array.isArray(value) || value.length < 3) return null
  const ring: Ring = []
  for (const point of value) {
    if (!Array.isArray(point) || point.length !== 2) return null
    const [longitude, latitude] = point
    if (
      typeof longitude !== 'number' ||
      typeof latitude !== 'number' ||
      !Number.isFinite(longitude) ||
      Math.abs(longitude) > 180 ||
      !Number.isFinite(latitude) ||
      Math.abs(latitude) > 90
    )
      return null
    ring.push([longitude, latitude])
  }
  return ring
}

/**
 * Parses `GET /api/landslides/{id}/geometry`, which returns a GeoJSON feature in
 * an envelope together with the source coordinate convention.
 */
export function parseSlideGeometry(payload: unknown): SlideGeometry | null {
  if (!payload || typeof payload !== 'object') return null
  const body = payload as Record<string, unknown>
  const feature = (body.feature ?? body) as Record<string, unknown>
  const geometry = feature.geometry as Record<string, unknown> | null | undefined
  if (!geometry || typeof geometry !== 'object') return null
  const type = geometry.type
  const coordinates = geometry.coordinates
  let ringValue: unknown
  if (type === 'Polygon' && Array.isArray(coordinates)) ringValue = coordinates[0]
  else if (type === 'MultiPolygon' && Array.isArray(coordinates)) {
    const first = coordinates[0]
    ringValue = Array.isArray(first) ? first[0] : undefined
  } else return null
  const ring = parseRing(ringValue)
  if (!ring) return null
  const coordinateSystem = body.coordinateSystem as Record<string, unknown> | undefined
  const properties = (feature.properties ?? {}) as Record<string, unknown>
  return {
    ring,
    canOverlay: body.canOverlay === true,
    kind: typeof properties.kind === 'string' ? properties.kind : 'polygon',
    latitudeConvention:
      typeof coordinateSystem?.latitude === 'string' ? coordinateSystem.latitude : 'unverified',
  }
}
