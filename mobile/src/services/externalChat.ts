import { fetchExternalChatContext } from './api'
import type { ChatMessage, ChatStreamHandlers, ExternalChatContext } from './api'
import { chatEventData } from './chatEvents'

export const modelPresets = {
  deepseek: {
    label: 'DeepSeek',
    endpoint: 'https://api.deepseek.com/chat/completions',
    model: 'deepseek-flash',
  },
  qwen: {
    label: '通义千问（百炼北京）',
    endpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    model: 'qwen-plus',
  },
  custom: { label: '自定义 API', endpoint: '', model: '' },
} as const

export type ExternalProvider = keyof typeof modelPresets

export interface ExternalModelSettings {
  provider: ExternalProvider
  endpoint: string
  model: string
}

export interface ExternalModel extends ExternalModelSettings {
  apiKey: string
}

function chatEndpoint(value: string): string {
  let url: URL
  try {
    url = new URL(value.trim())
  } catch {
    throw new Error('请输入有效的 HTTPS 接口地址。')
  }
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) {
    throw new Error('请输入 HTTPS 接口地址，密钥请填在 API Key 栏。')
  }
  const path = url.pathname.replace(/\/+$/, '')
  url.pathname = path.endsWith('/chat/completions') ? path : path + '/chat/completions'
  return url.href
}

export function prepareExternalModel(
  draft: ExternalModel,
  current: ExternalModel | null,
): ExternalModel {
  const endpoint = chatEndpoint(modelPresets[draft.provider].endpoint || draft.endpoint)
  const model = draft.model.trim()
  if (!model) throw new Error('请输入模型 ID。')
  const boundToCurrent = current?.provider === draft.provider && current.endpoint === endpoint
  const apiKey = draft.apiKey.trim() || (boundToCurrent ? current.apiKey : '')
  if (!apiKey) throw new Error('请输入这个接口对应的 API Key。')
  if (!/^[\x21-\x7e]+$/.test(apiKey)) throw new Error('API Key 格式有误，请重新粘贴。')
  return { provider: draft.provider, endpoint, model, apiKey }
}

type ProviderReply = {
  error?: unknown
  choices?: {
    message?: { content?: unknown }
    delta?: { content?: unknown }
    finish_reason?: string | null
  }[]
}

const systemPrompt =
  '你是火星滑坡助手。用中文回答。涉及本项目的分类、数量、坐标或影像时，以公开图鉴摘录和对话中查到的项目记录为准，资料不足时说明无法确认。'

function modelPrompt(context: ExternalChatContext | null) {
  const excerpts = context?.excerpts.map((item) => `${item.title}\n${item.text}`).join('\n\n')
  return excerpts ? `${systemPrompt}\n\n公开图鉴摘录\n${excerpts}` : systemPrompt
}

function providerError(status: number) {
  if (status === 401 || status === 403) return '接口拒绝访问。请检查 API Key 及模型权限。'
  if (status === 429) return '接口调用受限。请检查额度或稍后重试。'
  if (status === 400 || status === 404) return '接口无法处理请求。请检查接口地址和模型 ID。'
  return `模型接口请求失败（HTTP ${status}）。`
}

export async function streamExternalChat(
  model: ExternalModel,
  messages: ChatMessage[],
  handlers: ChatStreamHandlers,
) {
  const context = await fetchExternalChatContext(messages[messages.length - 1].content)
  if (context?.reply) {
    handlers.onDone?.(context.reply)
    return
  }

  let response: Response
  try {
    response = await fetch(model.endpoint, {
      method: 'POST',
      headers: { Authorization: `Bearer ${model.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model.model,
        messages: [{ role: 'system', content: modelPrompt(context) }, ...messages],
        stream: true,
      }),
      credentials: 'omit',
      redirect: 'error',
      cache: 'no-store',
      referrerPolicy: 'no-referrer',
      signal: AbortSignal.timeout(120000),
    })
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      throw new Error('模型接口响应超时，请稍后重试。')
    }
    throw new Error('无法连接模型接口。请检查地址及接口的跨域访问设置。')
  }
  if (!response.ok) throw new Error(providerError(response.status))

  let answer = ''
  try {
    if (response.headers.get('content-type')?.includes('text/event-stream') && response.body) {
      let finished = false
      for await (const data of chatEventData(response.body)) {
        if (data.trim() === '[DONE]') {
          finished = true
          break
        }
        const frame = JSON.parse(data) as ProviderReply
        if (frame.error) throw new Error()
        const choice = frame.choices?.[0]
        if (typeof choice?.delta?.content === 'string') {
          answer += choice.delta.content
          handlers.onDelta?.(choice.delta.content)
        }
        if (choice?.finish_reason) finished = true
      }
      if (!finished) throw new Error()
    } else {
      const data = (await response.json()) as ProviderReply
      const content = data.choices?.[0]?.message?.content
      if (!data.error && typeof content === 'string') answer = content
    }
  } catch {
    throw new Error('模型接口的回复中断或格式不兼容，请重试。')
  }
  if (!answer.trim()) throw new Error('模型没有返回可显示的文字。请检查模型是否支持文本对话。')
  handlers.onDone?.({
    status: 'ok',
    answer,
    records: [],
    citations: context?.citations ?? [],
    actions: [],
    toolCalls: [],
  })
}
