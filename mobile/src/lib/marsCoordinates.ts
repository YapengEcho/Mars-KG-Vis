export interface MarsCoordinate {
  longitude: number
  latitude: number
}

/** East-positive longitude, normalized to [-180, 180). */
export function normalizeLongitude(longitude: number): number {
  return ((((longitude + 180) % 360) + 360) % 360) - 180
}

export function isMarsCoordinate(coordinate: MarsCoordinate): boolean {
  return (
    Number.isFinite(coordinate.longitude) &&
    Number.isFinite(coordinate.latitude) &&
    coordinate.latitude >= -90 &&
    coordinate.latitude <= 90
  )
}

/** The prime meridian faces +Z; the north pole points toward +Y. */
export function lonLatToCartesian(
  longitude: number,
  latitude: number,
  radius = 1,
): [number, number, number] {
  const lon = (longitude * Math.PI) / 180
  const lat = (latitude * Math.PI) / 180
  const equatorialRadius = radius * Math.cos(lat)
  return [
    equatorialRadius * Math.sin(lon),
    radius * Math.sin(lat),
    equatorialRadius * Math.cos(lon),
  ]
}

export function cartesianToLonLat(x: number, y: number, z: number): MarsCoordinate {
  const radius = Math.hypot(x, y, z)
  if (radius === 0) return { longitude: 0, latitude: 0 }
  return {
    longitude: normalizeLongitude((Math.atan2(x, z) * 180) / Math.PI),
    latitude: (Math.asin(Math.max(-1, Math.min(1, y / radius))) * 180) / Math.PI,
  }
}

export function formatMarsCoordinate({ longitude, latitude }: MarsCoordinate): string {
  const lon = normalizeLongitude(longitude)
  return `${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'} · ${Math.abs(latitude).toFixed(2)}°${latitude < 0 ? 'S' : 'N'}`
}
