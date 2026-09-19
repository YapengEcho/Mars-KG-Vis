// mock 数据 —— 字段必须与 docs/03-里程碑与接口约定.md 完全一致
// 开发阶段使用；后端就绪后切换 src/api/index.js 的 USE_MOCK

// GET /api/landslides/  —— results[] 最少字段
export const landslidesMock = {
  count: 6,
  page: 1,
  page_size: 20,
  results: [
    {
      id: 'ls_0001',
      name: 'example crater wall',
      type: 'slump',
      lat: 12.3,
      lon: -45.6,
      length_m: 1200,
      mission: 'MRO',
    },
    {
      id: 'ls_0002',
      name: 'aurora camp scarp',
      type: 'fall',
      lat: 20.5,
      lon: -30.2,
      length_m: 640,
      mission: 'MRO',
    },
    {
      id: 'ls_0003',
      name: 'eos chasma block',
      type: 'flow',
      lat: -14.8,
      lon: -58.1,
      length_m: 2100,
      mission: 'HIRISE',
    },
    {
      id: 'ls_0004',
      name: 'hebes rim collapse',
      type: 'slump',
      lat: -1.2,
      lon: -76.6,
      length_m: 980,
      mission: 'MRO',
    },
    {
      id: 'ls_0005',
      name: 'north polar slide',
      type: 'flow',
      lat: 70.1,
      lon: 138.4,
      length_m: 3600,
      mission: 'MRO',
    },
    {
      id: 'ls_0006',
      name: 'tharsis gully',
      type: 'fall',
      lat: 25.4,
      lon: -92.8,
      length_m: 420,
      mission: 'VO',
    },
  ],
}

// GET /api/stats/overview/
export const statsMock = {
  total: 6,
  by_type: [
    { type: 'slump', count: 2 },
    { type: 'fall', count: 2 },
    { type: 'flow', count: 2 },
  ],
  by_mission: [
    { mission: 'MRO', count: 4 },
    { mission: 'HIRISE', count: 1 },
    { mission: 'VO', count: 1 },
  ],
}

// GET /api/landslides/{id}/
export const landslideDetailMock = (id) => ({
  id,
  name: 'example crater wall',
  type: 'slump',
  lat: 12.3,
  lon: -45.6,
  length_m: 1200,
  width_m: 380,
  mission: 'MRO',
  features: ['scarp', 'debris apron'],
  description: '沿坑壁发生的崩滑型滑坡，呈弧形后缘与舌状堆积体。',
  source_url: 'https://example.com/landslides/ls_0001',
})

// GET /api/graph/subgraph/
export const subgraphMock = {
  degraded: false,
  nodes: [
    { id: 'ls_0001', label: 'example crater wall', kind: 'Landslide' },
    { id: 'type_slump', label: 'slump', kind: 'LandslideType' },
    { id: 'reg_valles', label: 'Valles Marineris', kind: 'Region' },
    { id: 'feat_scarp', label: 'scarp', kind: 'Feature' },
  ],
  edges: [
    { source: 'ls_0001', target: 'type_slump', rel: 'IS_TYPE' },
    { source: 'ls_0001', target: 'reg_valles', rel: 'LOCATED_IN' },
    { source: 'ls_0001', target: 'feat_scarp', rel: 'HAS_FEATURE' },
  ],
}

// GET /api/recommend/ —— items[] 字段
export const recommendMock = {
  items: [
    { id: 'ls_0004', score: 0.83, type: 'slump' },
    { id: 'ls_0002', score: 0.61, type: 'fall' },
    { id: 'ls_0003', score: 0.45, type: 'flow' },
  ],
}

// POST /api/predict/
export const predictMock = {
  pred_type: 'slump',
  proba: { slump: 0.72, fall: 0.21, flow: 0.07 },
}

// POST /api/agent/ask/ —— answer 字段
export const agentMock = {
  answer: '火星上常见的滑坡类型主要包括 slump（崩滑）、fall（坠落）和 flow（流动）三类。',
  citations: [
    { id: 'ls_0001', title: 'example crater wall' },
    { id: 'ls_0003', title: 'eos chasma block' },
  ],
  degraded: false,
}

export default function resolveMock(method, url) {
  // 统一入口：去掉可选前缀 /api，与前端请求路径对齐
  const p = url.replace(/^\/+api\//, '/')
  if (method === 'GET' && p.startsWith('/landslides/')) {
    // 列表查询带 query（?page=...），详情路径形如 /landslides/{id}/
    const detailOk = /^\/landslides\/[^/?]+\/?$/.test(p) && !p.includes('?')
    if (!detailOk) return { ...landslidesMock }
    const id = p.replace(/^\/landslides\//, '').replace(/\/$/, '')
    return landslideDetailMock(id)
  }
  if (method === 'GET' && p.startsWith('/stats/')) return { ...statsMock }
  if (method === 'GET' && p.startsWith('/graph/')) return { ...subgraphMock }
  if (method === 'GET' && p.startsWith('/recommend/')) return { ...recommendMock }
  if (method === 'POST' && p.startsWith('/predict/')) return { ...predictMock }
  if (method === 'POST' && p.startsWith('/agent/')) return { ...agentMock }
  return {}
}