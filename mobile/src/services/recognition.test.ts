import { beforeEach, describe, expect, it, vi } from 'vitest'
vi.mock('./api', () => ({ apiBase: 'http://127.0.0.1:8001' }))
import {
  submitRecognition,
  submitDetection,
  recognitionHistory,
  recognitionImage,
  getRecognition,
  deleteRecognition,
  resetRecognitionSession,
  taskLabel,
  type RecognitionTask,
  type DetectionTask,
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

describe('two-stage recognition outcomes', () => {
  const completed: RecognitionTask = {
    taskId: 'task-1',
    kind: 'classify',
    status: 'completed',
    createdAt: '2026-01-01T00:00:00Z',
    finishedAt: '2026-01-01T00:00:01Z',
    modelVersion: 'type-model',
    preprocessingVersion: 'v1',
    errorCode: null,
    result: null,
  }

  it('labels each initial screening outcome', () => {
    const outcomes = [
      ['possible_landslide', '疑似滑坡 · 待复核'],
      ['no_landslide', '未检出明显滑坡'],
      ['unsupported_image', '影像不适用'],
      ['inconclusive', '初筛结果不确定'],
    ] as const
    for (const [decision, label] of outcomes) {
      expect(
        taskLabel({
          ...completed,
          screening: { decision, score: null, threshold: null, modelVersion: null },
        }),
      ).toBe(label)
    }
  })

  it('does not treat historical type-only results as confirmed slides', () => {
    expect(
      taskLabel({
        ...completed,
        result: {
          prediction: 'CL',
          probabilities: { SL: 0.1, SP: 0.1, FL: 0.1, CL: 0.7 },
          confidence: 0.7,
          uncertain: true,
          reviewReasons: ['MODEL_REQUIRES_HUMAN_REVIEW'],
          scoresCalibrated: false,
          modelVersion: 'type-model',
          preprocessingVersion: 'v1',
        },
      }),
    ).toBe('旧记录 · 未初筛')
  })
})
describe('detection tasks', () => {
  const pending: DetectionTask = {
    taskId: 'task-detect',
    kind: 'detect',
    status: 'queued',
    createdAt: '2026-01-01T00:00:00Z',
    finishedAt: null,
    modelVersion: 'detector-v1',
    preprocessingVersion: 'detect-v1',
    input: { width: 1800, height: 1500, preprocessingVersion: 'detect-v1' },
    result: null,
    errorCode: null,
    imageUrl: '/api/recognition/tasks/task-detect/image',
  }
  const candidate = {
    candidateId: 'c1',
    box: [10, 20, 110, 120] as [number, number, number, number],
    detectionScore: 0.4,
    classification: {
      prediction: 'CL' as const,
      probabilities: { SL: 0.1, SP: 0.2, FL: 0.3, CL: 0.4 },
      confidence: 0.4,
      uncertain: true,
      reviewReasons: [],
      scoresCalibrated: false,
      modelVersion: 'type-model',
      preprocessingVersion: 'v1',
    },
  }
  it('uploads to the detection endpoint and lists detect history', async () => {
    fetcher
      .mockResolvedValueOnce(response(credential, 201))
      .mockResolvedValueOnce(response({ taskId: 'task-detect', kind: 'detect' }, 202))
    expect((await submitDetection(new File(['x'], 'ctx.png'))).taskId).toBe('task-detect')
    expect(fetcher.mock.calls[1][0]).toContain('/api/landslide/detect')
    fetcher.mockResolvedValueOnce(response({ items: [], total: 0 }))
    await recognitionHistory(1, 'detect')
    expect(fetcher.mock.calls[2][0]).toContain('kind=detect')
  })
  it('labels pending, empty, and candidate outcomes', () => {
    expect(taskLabel({ ...pending, status: 'running' })).toBe('正在检测')
    expect(
      taskLabel({
        ...pending,
        status: 'completed',
        result: {
          outcome: 'no_detections',
          coordinateSpace: 'image_pixels',
          truncated: false,
          candidates: [],
        },
      }),
    ).toBe('未检出候选 · 待复核')
    expect(
      taskLabel({
        ...pending,
        status: 'completed',
        result: {
          outcome: 'detections',
          coordinateSpace: 'image_pixels',
          truncated: false,
          candidates: [candidate, { ...candidate, candidateId: 'c2' }],
        },
      }),
    ).toBe('检出 2 个候选 · 待复核')
  })
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
  it('maps a network failure to a retryable message', async () => {
    fetcher.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await expect(submitRecognition(new File(['x'], 'x.png'))).rejects.toThrow(
      '无法连接识别服务',
    )
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
