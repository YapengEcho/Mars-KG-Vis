import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('./api', () => ({ apiBase: 'http://127.0.0.1:8001' }))
import {
  submitRecognition,
  recognitionHistory,
  recognitionImage,
  getRecognition,
  deleteRecognition,
  resetRecognitionSession,
} from './recognition'
const memory = new Map<string, string>()
const storage = {
  getItem: vi.fn((key: string) => memory.get(key) ?? null),
  setItem: vi.fn((key: string, value: string) => {
    memory.set(key, value)
  }),
  removeItem: vi.fn((key: string) => {
    memory.delete(key)
  }),
}
const fetcher = vi.fn()
const credential = { token: 'test-device-session', expiresAt: '2099-01-01T00:00:00Z' }
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
beforeEach(() => {
  memory.clear()
  vi.clearAllMocks()
  vi.stubGlobal('sessionStorage', storage)
  vi.stubGlobal('fetch', fetcher)
})
describe('device-scoped recognition', () => {
  it('retains one session, submits multipart, and uses authenticated reads and deletes', async () => {
    fetcher
      .mockResolvedValueOnce(response(credential, 201))
      .mockResolvedValueOnce(response({ taskId: 'task-1', status: 'queued' }, 202))
    const file = new File(['test'], 'test.png', { type: 'image/png' })
    expect((await submitRecognition(file, [10, 20, 200, 210])).taskId).toBe('task-1')
    expect(fetcher.mock.calls[1][1].body.get('file')).toBe(file)
    expect(fetcher.mock.calls[1][1].body.get('roi')).toBe('[10,20,200,210]')
    fetcher
      .mockResolvedValueOnce(response({ items: [], total: 0 }))
      .mockResolvedValueOnce(response({ taskId: 'task-1' }))
      .mockResolvedValueOnce(new Response('image'))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    await recognitionHistory()
    await getRecognition('task-1')
    await recognitionImage('task-1')
    await deleteRecognition('task-1')
    for (const [url, init] of fetcher.mock.calls.slice(1)) {
      expect(url).not.toContain(credential.token)
      expect(init.headers.Authorization).toBe('Bearer ' + credential.token)
      expect(init.cache).toBe('no-store')
    }
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/sessions'))).toHaveLength(1)
  })
  it('does not create empty sessions just to show history', async () => {
    expect(await recognitionHistory()).toEqual({ items: [], total: 0 })
    expect(fetcher).not.toHaveBeenCalled()
  })
  it('refuses to upload when the credential cannot be saved', async () => {
    fetcher.mockResolvedValueOnce(response(credential, 201))
    storage.setItem.mockImplementationOnce(() => {
      throw new Error('storage blocked')
    })
    await expect(submitRecognition(new File(['x'], 'x.png'))).rejects.toThrow('图片未提交')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
  it('does not silently replace a rejected credential or lose its history', async () => {
    memory.set(
      'mars-atlas:recognition-session:v1:http://127.0.0.1:8001',
      JSON.stringify(credential),
    )
    fetcher.mockResolvedValueOnce(response({ detail: { code: 'AUTH_REQUIRED' } }, 401))
    await expect(recognitionHistory()).rejects.toThrow('凭据已失效')
    expect(memory.size).toBe(1)
    expect(fetcher).toHaveBeenCalledTimes(1)
    resetRecognitionSession()
    expect(memory.size).toBe(0)
  })
  it('does not issue a new session after expiry until explicitly reset', async () => {
    memory.set(
      'mars-atlas:recognition-session:v1:http://127.0.0.1:8001',
      JSON.stringify({ ...credential, expiresAt: '2000-01-01T00:00:00Z' }),
    )
    await expect(recognitionHistory()).rejects.toThrow('凭据已失效')
    expect(fetcher).not.toHaveBeenCalled()
  })
  it('maps a busy queue to a useful error and never retries uploads', async () => {
    fetcher
      .mockResolvedValueOnce(response(credential, 201))
      .mockResolvedValueOnce(response({ detail: { code: 'QUEUE_FULL' } }, 429))
    await expect(submitRecognition(new File(['x'], 'x.png'))).rejects.toThrow('稍后再试')
    expect(fetcher).toHaveBeenCalledTimes(2)
  })
})
