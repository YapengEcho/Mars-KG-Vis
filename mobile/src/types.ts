export const slideTypes = ['SL', 'SP', 'FL', 'CL'] as const
export type SlideType = (typeof slideTypes)[number]
export interface Slide {
  id: string
  sourceId: number | null
  name: string
  type: SlideType
  longitude: number
  latitude: number
  areaKm2: number | null
  summary: string
  imageUrl: string | null
  source: { title: string; url: string | null }
  isDemo: boolean
}
export interface Catalog {
  version: string
  mode: 'demo' | 'catalog'
  coordinateSystem: { body: 'Mars'; longitude: string; latitude: string }
  records: Slide[]
}
export const typeInfo: Record<
  SlideType,
  { name: string; english: string; color: string; note: string }
> = {
  SL: {
    name: '滑移型',
    english: 'Sliding',
    color: '#ec9c79',
    note: '按原编目中的 SL 类别查看记录与案例。',
  },
  SP: {
    name: '滑塌型',
    english: 'Slumping',
    color: '#78bed4',
    note: '按原编目中的 SP 类别查看记录与案例。',
  },
  FL: {
    name: '流动型',
    english: 'Flowing',
    color: '#b9a2e8',
    note: '按原编目中的 FL 类别查看记录与案例。',
  },
  CL: {
    name: '复合型',
    english: 'Composite',
    color: '#e1c574',
    note: '按原编目中的 CL 类别查看记录与案例。',
  },
}
export function coordinateLabel(lon: number, lat: number): string {
  return `${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'} · ${Math.abs(lat).toFixed(2)}°${lat < 0 ? 'S' : 'N'}`
}
