import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatResponse } from '../services/api'

const streamChatMessage = vi.fn()
const recognitionAuthHeaders = vi.fn()

vi.mock('../services/api', () => ({
  streamChatMessage: (...args: unknown[]) => streamChatMessage(...args),
}))

vi.mock('../services/recognition', () => ({
  recognitionAuthHeaders: () => recognitionAuthHeaders(),
}))

// Imported after the mock so the store picks up the stubbed transport.
const { chat } = await import('./chat')

type Handlers = {
  onDelta?: (chunk: string) => void
  onDone?: (response: ChatResponse) => void
}

function reply(answer: string): ChatResponse {
  return {
    status: 'ok',
    answer,
    records: [],
    citations: [],
    actions: [],
    toolCalls: [],
  }
}

/** Messages payload the store sent for a given request index. */
function sentMessages(index: number) {
  return streamChatMessage.mock.calls[index][0] as { role: string; content: string }[]
}

describe('chat store streaming history', () => {
  beforeEach(() => {
    chat.reset()
    streamChatMessage.mockReset()
  })

  it('omits a reply that was cut off before the done frame', async () => {
    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDelta?.('SP 总数是三')
      // Stream ends without a done frame, simulating a dropped connection.
    })

    await chat.send('SP 有多少')
    expect(chat.state.error).toBe('回复未完成，请重试')
    expect(chat.state.thread.at(-1)).toMatchObject({
      role: 'assistant',
      content: 'SP 总数是三',
      pending: true,
    })

    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDone?.(reply('共 3 条'))
    })

    await chat.send('重新回答')

    // The truncated answer stays visible but must not be replayed to the model.
    expect(chat.state.thread.some((entry) => entry.content === 'SP 总数是三')).toBe(true)
    expect(sentMessages(1)).toEqual([
      { role: 'user', content: 'SP 有多少' },
      { role: 'user', content: '重新回答' },
    ])
  })

  it('keeps a completed reply in the following request', async () => {
    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDelta?.('共 3 条')
      handlers.onDone?.(reply('共 3 条'))
    })

    await chat.send('SP 有多少')
    expect(chat.state.thread.at(-1)?.pending).toBe(false)
    expect(chat.state.error).toBe('')

    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDone?.(reply('好的'))
    })

    await chat.send('继续')

    expect(sentMessages(1)).toEqual([
      { role: 'user', content: 'SP 有多少' },
      { role: 'assistant', content: '共 3 条' },
      { role: 'user', content: '继续' },
    ])
  })

  it('treats a transport error after partial output as incomplete', async () => {
    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDelta?.('部分内容')
      throw new Error('网络中断')
    })

    await chat.send('问题')

    expect(chat.state.error).toBe('网络中断')
    expect(chat.state.thread.at(-1)).toMatchObject({ content: '部分内容', pending: true })
    expect(chat.state.busy).toBe(false)
  })
})

describe('chat store selected recognition task', () => {
  const taskId = 'a'.repeat(32)

  beforeEach(() => {
    chat.reset()
    streamChatMessage.mockReset()
    recognitionAuthHeaders.mockReset()
  })

  it('forwards the task id and device credential with the request', async () => {
    recognitionAuthHeaders.mockResolvedValueOnce({ Authorization: 'Bearer device-token' })
    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDone?.(reply('本次检测找到 1 个候选框，待复核。'))
    })

    await chat.send('解释这条记录', taskId)

    const [, , , task] = streamChatMessage.mock.calls[0]
    expect(task).toEqual({
      recognitionTaskId: taskId,
      headers: { Authorization: 'Bearer device-token' },
    })
  })

  it('refuses to send when the device has no credential', async () => {
    recognitionAuthHeaders.mockResolvedValueOnce(null)

    const sent = await chat.send('解释这条记录', taskId)

    expect(sent).toBe(false)
    expect(streamChatMessage).not.toHaveBeenCalled()
    expect(chat.state.error).toContain('识别记录')
    // 没有发出去的问题不能留在对话里。
    expect(chat.state.thread.some((entry) => entry.role === 'user')).toBe(false)
  })

  it('records the task kind so 打开原记录 can pick detection mode', async () => {
    recognitionAuthHeaders.mockResolvedValueOnce({ Authorization: 'Bearer device-token' })
    streamChatMessage.mockImplementationOnce(async (_messages, handlers: Handlers) => {
      handlers.onDone?.({
        ...reply('本次检测运行完成，未检出。'),
        recognitionTask: { kind: 'detect', status: 'completed' },
      })
    })

    await chat.send('解释这条记录', taskId)

    expect(chat.state.thread.at(-1)?.taskKind).toBe('detect')
  })
})
