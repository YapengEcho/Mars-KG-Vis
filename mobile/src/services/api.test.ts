import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The SSE parser is replaced by a single terminal frame so these tests focus on
 * the request the chat transport builds, not on stream decoding.
 */
vi.mock('./chatEvents', () => ({
  chatEventData: async function* () {
    yield JSON.stringify({
      type: 'done',
      response: { status: 'ok', answer: '好的', records: [], citations: [], actions: [], toolCalls: [] },
    })
  },
}))

const fetchMock = vi.fn()
vi.stubGlobal('fetch', fetchMock)

const TASK_ID = 'a'.repeat(32)
const CREDENTIAL = { Authorization: 'Bearer device-token' }

describe('chat transport with a selected recognition task', () => {
  let api: typeof import('./api')

  beforeAll(async () => {
    // apiBase is captured at module load, so the env must be set before import.
    vi.stubEnv('VITE_API_BASE_URL', 'http://test.local')
    api = await import('./api')
  })

  beforeEach(() => {
    fetchMock.mockReset()
    fetchMock.mockResolvedValue(new Response('', { status: 200 }))
  })

  it('sends the task id and the device credential', async () => {
    const done = vi.fn()

    await api.streamChatMessage([{ role: 'user', content: '解释这条记录' }], { onDone: done }, undefined, {
      recognitionTaskId: TASK_ID,
      headers: CREDENTIAL,
    })

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('http://test.local/api/agent/chat/stream')
    expect(init.headers).toMatchObject({
      'Content-Type': 'application/json',
      Authorization: 'Bearer device-token',
    })
    expect(JSON.parse(String(init.body))).toMatchObject({ recognitionTaskId: TASK_ID })
    expect(done).toHaveBeenCalled()
  })

  it('keeps ordinary chat free of task fields and credentials', async () => {
    await api.streamChatMessage([{ role: 'user', content: '滑移型是什么' }], {})

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(init.headers).not.toHaveProperty('Authorization')
    expect(JSON.parse(String(init.body))).not.toHaveProperty('recognitionTaskId')
  })

  it('explains an expired credential instead of reporting an outage', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: { code: 'AUTH_REQUIRED' } }), { status: 401 }),
    )

    await expect(
      api.streamChatMessage([{ role: 'user', content: '解释' }], {}, undefined, {
        recognitionTaskId: TASK_ID,
        headers: CREDENTIAL,
      }),
    ).rejects.toThrow('识别凭据已失效')
  })

  it('explains a task that belongs to another device', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: { code: 'TASK_NOT_FOUND' } }), { status: 404 }),
    )

    await expect(
      api.streamChatMessage([{ role: 'user', content: '解释' }], {}, undefined, {
        recognitionTaskId: TASK_ID,
        headers: CREDENTIAL,
      }),
    ).rejects.toThrow('不属于当前设备')
  })
})
