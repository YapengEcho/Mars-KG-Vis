const configured = import.meta.env.VITE_API_BASE_URL?.trim()
export const apiBase = configured ? configured.replace(/\/$/, '') : ''
export interface Capabilities {
  classifier: { available: boolean; reason?: string }
  detection: { available: boolean; reason?: string }
}
export async function fetchCapabilities(): Promise<Capabilities | null> {
  if (!apiBase) return null
  const r = await fetch(`${apiBase}/api/capabilities`, { signal: AbortSignal.timeout(8000) })
  if (!r.ok) throw new Error('暂时无法连接识别服务')
  return r.json() as Promise<Capabilities>
}
export async function classifyImage(file: File): Promise<unknown> {
  if (!apiBase) throw new Error('识别服务尚未接入')
  const body = new FormData()
  body.append('file', file)
  const r = await fetch(`${apiBase}/api/landslide/classify`, {
    method: 'POST',
    body,
    signal: AbortSignal.timeout(60000),
  })
  if (!r.ok) throw new Error(r.status === 503 ? '识别服务尚未就绪' : '识别失败，请稍后重试')
  return r.json()
}
