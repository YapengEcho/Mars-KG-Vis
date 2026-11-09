import { describe, expect, it } from 'vitest'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { createMemoryHistory, createRouter } from 'vue-router'
import AtlasView from './AtlasView.vue'
import AtlasTypeView from './AtlasTypeView.vue'
import AtlasCompareView from './AtlasCompareView.vue'
import AtlasCaseView from './AtlasCaseView.vue'

const routes = [
  { path: '/atlas', component: AtlasView },
  { path: '/atlas/compare', component: AtlasCompareView },
  { path: '/atlas/type/:type', component: AtlasTypeView, props: true },
  { path: '/atlas/case/:id', component: AtlasCaseView, props: true },
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

describe('atlas pages render without the private catalogue', () => {
  it('lists the four types and twelve cases', async () => {
    const html = await render('/atlas')
    expect(html).toContain('滑坡类型与目录')
    expect(html).toContain('/atlas/type/SL')
    expect(html).toContain('12 个案例')
    expect(html.match(/case-ls-/g)?.length).toBeGreaterThanOrEqual(12)
  })
  it('explains one type with its cases and citations', async () => {
    const html = await render('/atlas/type/SL')
    expect(html).toContain('滑移型')
    expect(html).toContain('表 2')
    expect(html).toContain('远程堆积的滑移型')
  })
  it('renders both comparisons as tables', async () => {
    const html = await render('/atlas/compare')
    expect(html).toContain('SL 滑移型 vs SP 滑塌型')
    expect(html.match(/<table/g)?.length).toBe(2)
  })
  it('explains a missing record instead of inventing coordinates', async () => {
    const html = await render('/atlas/case/case-ls-0493')
    expect(html).toContain('远程堆积的滑移型')
    expect(html).toContain('当前目录里没有')
    expect(html).not.toMatch(/°[EW]/)
  })
  it('rejects an unknown type politely', async () => {
    const html = await render('/atlas/type/XX')
    expect(html).toContain('未知的滑坡类别')
  })
})
