import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from './api'
import { prepareExternalModel, streamExternalChat } from './externalChat'

const { fetchExternalChatContext } = vi.hoisted(() => ({ fetchExternalChatContext: vi.fn() }))
vi.mock('./api', () => ({ fetchExternalChatContext }))

const model = {
  provider: 'custom' as const,
  model: 'fixture-model',
  endpoint: 'https://model.example/v1/chat/completions',
  apiKey: 'fixture-only',
}
const messages: ChatMessage[] = [{ role: 'user', content: '解释滑坡分类' }]
const fetchModel = vi.fn()

beforeEach(() => {
  fetchModel.mockReset()
  fetchExternalChatContext.mockReset().mockResolvedValue(null)
  vi.stubGlobal('fetch', fetchModel)
})
afterEach(() => vi.unstubAllGlobals())

describe('external model configuration', () => {
  it('normalizes base URLs and prevents carrying a key across endpoints', () => {
    const draft = { ...model, endpoint: 'https://model.example/v1/', apiKey: '' }
    expect(prepareExternalModel(draft, model)).toEqual(model)
    expect(() =>
      prepareExternalModel({ ...draft, endpoint: 'https://different.example/v1' }, model),
    ).toThrow('API Key')
    expect(() =>
      prepareExternalModel({ ...model, endpoint: 'http://model.example/v1' }, null),
    ).toThrow('HTTPS')
  })

  it('calls the selected HTTPS endpoint with the credential only in its auth header', async () => {
    fetchModel.mockResolvedValue(
      Response.json({ choices: [{ message: { content: '来自外部模型' } }] }),
    )
    const onDone = vi.fn()
    await streamExternalChat(model, messages, { onDone })
    expect(fetchExternalChatContext).toHaveBeenCalledWith(messages[0].content)
    const [url, options] = fetchModel.mock.calls[0]
    expect(url).toBe(model.endpoint)
    expect(options.headers.Authorization).toBe('Bearer fixture-only')
    expect(options.redirect).toBe('error')
    expect(options.body).not.toContain(model.apiKey)
    expect(JSON.parse(options.body)).toMatchObject({ model: model.model, stream: true })
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ answer: '来自外部模型' }))
  })

  it('streams CRLF frames and validates the terminal event', async () => {
    fetchModel.mockResolvedValue(
      new Response(
        'data: {"choices":[{"delta":{"content":"滑坡"}}]}\r\n\r\n' +
          'data: {"choices":[{"delta":{"content":"类型"},"finish_reason":"stop"}]}\r\n\r\n' +
          'data: [DONE]\r\n\r\n',
        { headers: { 'Content-Type': 'text/event-stream' } },
      ),
    )
    const onDelta = vi.fn()
    const onDone = vi.fn()
    await streamExternalChat(model, messages, { onDelta, onDone })
    expect(onDelta.mock.calls).toEqual([['滑坡'], ['类型']])
    expect(onDone).toHaveBeenCalledWith(expect.objectContaining({ answer: '滑坡类型' }))
  })

  it('preserves local catalogue results without calling the external model', async () => {
    const reply = { answer: '已找到', actions: [{ type: 'locate', slideId: 'ls_0493' }] }
    fetchExternalChatContext.mockResolvedValue({ reply })
    const onDone = vi.fn()
    await streamExternalChat(model, messages, { onDone })
    expect(fetchModel).not.toHaveBeenCalled()
    expect(onDone).toHaveBeenCalledWith(reply)
  })

  it('reports a rejected key without relaying the provider response body', async () => {
    fetchModel.mockResolvedValue(new Response('fixture-only', { status: 401 }))
    await expect(streamExternalChat(model, messages, {})).rejects.toThrow('接口拒绝访问')
  })
})
