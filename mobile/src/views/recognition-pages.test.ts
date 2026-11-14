import { beforeEach, describe, expect, it } from 'vitest'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { createMemoryHistory, createRouter } from 'vue-router'
import MineView from './MineView.vue'
import RecognitionDetailView from './RecognitionDetailView.vue'
import { recognitions } from '../stores/recognitions'

const routes = [
  { path: '/mine', component: MineView },
  { path: '/mine/recognitions/:id', component: RecognitionDetailView },
  { path: '/recognize', component: { template: '<p>recognize</p>' } },
  { path: '/atlas/type/:type', component: { template: '<p>atlas</p>' } },
  { path: '/', component: { template: '<p>explore</p>' } },
]

async function render(path: string): Promise<string> {
  const router = createRouter({ history: createMemoryHistory(), routes })
  const app = createSSRApp({ template: '<RouterView />' })
  app.use(router)
  await router.push(path)
  await router.isReady()
  return renderToString(app)
}

const record = {
  id: 'rec-1',
  createdAt: '2026-11-14T02:30:00.000Z',
  fileName: 'crop.png',
  image: '',
  prediction: 'CL' as const,
  probabilities: { SL: 0.05, SP: 0.12, FL: 0.21, CL: 0.62 },
  confidence: 0.62,
  uncertain: true,
  modelVersion: 'dinov2-base-lora-r4',
}

describe('recognition record pages', () => {
  beforeEach(() => {
    recognitions.state.records = [record]
    recognitions.state.storageError = ''
  })
  it('lists saved recognitions on the mine page', async () => {
    const html = await render('/mine')
    expect(html).toContain('识别记录')
    expect(html).toContain('复合型 · CL')
    expect(html).toContain('/mine/recognitions/rec-1')
    expect(html).toContain('待复核')
  })
  it('renders one record for review with its type link', async () => {
    const html = await render('/mine/recognitions/rec-1')
    expect(html).toContain('置信度 62.0%')
    expect(html).toContain('dinov2-base-lora-r4')
    expect(html).toContain('/atlas/type/CL')
    expect(html).toContain('原图文件名：crop.png')
  })
  it('handles a record that no longer exists', async () => {
    const html = await render('/mine/recognitions/rec-2')
    expect(html).toContain('记录不存在')
  })
  it('shows an empty state before the first recognition', async () => {
    recognitions.state.records = []
    const html = await render('/mine')
    expect(html).toContain('还没有识别记录')
    expect(html).toContain('/recognize')
  })
})
