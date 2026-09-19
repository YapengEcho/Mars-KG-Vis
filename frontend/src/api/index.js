// API 客户端：封装对 Django 后端的请求
// USE_MOCK=true 时返回本地 mock（字段与 docs/03 一致）；=false 走真实 /api 代理
import resolveMock from '@/utils/mock'

const USE_MOCK = true
const BASE = '/api'

function delay(ms = 200) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function request(method, url, data) {
  if (USE_MOCK) {
    await delay()
    const resolved = resolveMock(method, url)
    if (method === 'POST') return resolved
    return resolved
  }
  const res = await fetch(`${BASE}${url}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: data ? JSON.stringify(data) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail || detail
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

// 采用按领域拆分的语义化方法，前端不直接拼 URL
export const api = {
  // 滑坡列表/详情
  listLandslides: (params = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== '')
    )
    return request('GET', `/landslides/?${q.toString()}`)
  },
  getLandslide: (id) => request('GET', `/landslides/${id}/`),
  // 大屏统计
  getStats: () => request('GET', '/stats/overview/'),
  // 图谱子图
  getSubgraph: (id, depth = 1) =>
    request('GET', `/graph/subgraph/?id=${id}&depth=${depth}`),
  // 预测 / 推荐
  predict: (payload) => request('POST', '/predict/', payload),
  recommend: (id, topK = 5) =>
    request('GET', `/recommend/?id=${id}&top_k=${topK}`),
  // RAG 智能体
  ask: (question, topK = 5) =>
    request('POST', '/agent/ask/', { question, top_k: topK }),
}

export default api