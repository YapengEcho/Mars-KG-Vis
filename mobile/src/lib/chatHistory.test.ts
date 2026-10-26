import { describe, expect, it } from 'vitest'
import { HISTORY_WINDOW, MAX_HISTORY_MESSAGES, buildChatHistory } from './chatHistory'

function threadOf(turns: number) {
  const thread: { role: 'user' | 'assistant'; content: string }[] = [
    { role: 'assistant', content: '你好，我是火星滑坡助手。' },
  ]
  for (let index = 1; index <= turns; index += 1) {
    thread.push({ role: 'user', content: `问题${index}` })
    thread.push({ role: 'assistant', content: `回答${index}` })
  }
  return thread
}

describe('buildChatHistory', () => {
  it('drops the leading welcome message so history starts with a user turn', () => {
    const history = buildChatHistory([
      { role: 'assistant', content: '欢迎' },
      { role: 'user', content: '问题1' },
    ])
    expect(history).toEqual([{ role: 'user', content: '问题1' }])
  })

  it('never exceeds the backend limit across long conversations', () => {
    for (const turns of [1, 12, 24, 25, 26, 40, 200]) {
      const history = buildChatHistory(threadOf(turns))
      expect(history.length).toBeGreaterThan(0)
      expect(history.length).toBeLessThanOrEqual(HISTORY_WINDOW)
      expect(history.length).toBeLessThanOrEqual(MAX_HISTORY_MESSAGES)
    }
  })

  it('keeps the most recent turn', () => {
    const history = buildChatHistory(threadOf(30))
    expect(history.at(-1)).toEqual({ role: 'assistant', content: '回答30' })
  })

  it('honours a custom window and clamps it to the backend limit', () => {
    expect(buildChatHistory(threadOf(5), 4).length).toBeLessThanOrEqual(4)
    expect(buildChatHistory(threadOf(60), 500).length).toBeLessThanOrEqual(MAX_HISTORY_MESSAGES)
  })

  it('skips blank messages', () => {
    const history = buildChatHistory([
      { role: 'user', content: '问题1' },
      { role: 'assistant', content: '   ' },
      { role: 'user', content: '问题2' },
    ])
    expect(history).toEqual([
      { role: 'user', content: '问题1' },
      { role: 'user', content: '问题2' },
    ])
  })
})
