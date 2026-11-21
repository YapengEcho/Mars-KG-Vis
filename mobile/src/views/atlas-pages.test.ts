import { describe, expect, it } from 'vitest'
import { createSSRApp } from 'vue'
import { renderToString } from 'vue/server-renderer'
import { createMemoryHistory, createRouter } from 'vue-router'
import AtlasView from './AtlasView.vue'
import AtlasTypeView from './AtlasTypeView.vue'
import AtlasCompareView from './AtlasCompareView.vue'
import AtlasCaseView from './AtlasCaseView.vue'
import CatalogRecordView from './CatalogRecordView.vue'

const routes = [
  { path: '/atlas', component: AtlasView },
  { path: '/atlas/compare', component: AtlasCompareView },
  { path: '/atlas/type/:type', component: AtlasTypeView, props: true },
  { path: '/atlas/case/:id', component: AtlasCaseView, props: true },
  { path: '/catalog/:id', component: CatalogRecordView, props: true },
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
    expect(html).toContain('滑坡图鉴')
    expect(html).toContain('12 个案例')
    expect(html).toContain('href="/atlas/compare"')
    for (const type of ['SL', 'SP', 'FL', 'CL']) {
      expect(html).toContain(`id="case-group-${type}"`)
      expect(html).toContain(`href="/atlas/type/${type}"`)
    }
    expect(html.match(/href="\/catalog\/ls_\d{4}\?from=atlas"/g)).toHaveLength(12)
    expect(html).not.toMatch(/<img\b|terrain-swatch|公开图集|\/atlas\/case\//)
  })
  it('puts recognition clues and direct case links before landform details', async () => {
    const html = await render('/atlas/type/FL')
    expect(html).toMatch(/<h1\b[^>]*>流动型<\/h1>/)
    expect(html).toContain('堆积体的前缘是否呈扇形或叶状')
    expect(html).toContain('表 2')
    for (const section of ['type-signals-title', 'type-cases-title', 'type-landforms-title'])
      expect(html).toContain(`id="${section}"`)
    expect(html.indexOf('id="type-signals-title"')).toBeLessThan(
      html.indexOf('id="type-cases-title"'),
    )
    expect(html.indexOf('id="type-cases-title"')).toBeLessThan(
      html.indexOf('id="type-landforms-title"'),
    )
    expect(html.match(/href="\/catalog\/ls_\d{4}\?from=atlas&amp;fromType=FL"/g)).toHaveLength(3)
    expect(html).not.toMatch(/<img\b|\/atlas\/case\//)
  })
  it('returns from a case to the type that opened it', async () => {
    const html = await render('/catalog/ls_0505?from=atlas&fromType=FL')
    expect(html).toContain('href="/atlas/type/FL"')
    expect(html).toContain('返回流动型')
  })
  it('renders one comparison at a time with navigable pair links', async () => {
    const html = await render('/atlas/compare')
    expect(html).toMatch(/<caption[^>]*>SL 滑移型 vs SP 滑塌型<\/caption>/)
    expect(html.match(/<table\b/g)).toHaveLength(1)
    expect(html).toContain('href="/atlas/compare?pair=sl-vs-sp"')
    expect(html).toContain('href="/atlas/compare?pair=sp-vs-cl"')

    const second = await render('/atlas/compare?pair=sp-vs-cl')
    expect(second).toMatch(/<caption[^>]*>SP 滑塌型 vs CL 复合型<\/caption>/)
    expect(second).toContain('下部结构')
    expect(second).not.toContain('运动距离')

    const unknown = await render('/atlas/compare?pair=unknown')
    expect(unknown).toContain('SL 滑移型 vs SP 滑塌型')
  })
  it('links from a type to a pair containing that type when it exists', async () => {
    const html = await render('/atlas/type/CL')
    expect(html).toContain('href="/atlas/compare?pair=sp-vs-cl"')
  })
  it('puts the image before case information and links the reference paper', async () => {
    const html = await render('/atlas/case/case-ls-0493')
    expect(html).toContain('远程堆积的滑移型')
    expect(html).toContain('ls_0493')
    expect(html).not.toContain('数据来源')
    expect(html).not.toContain('href="https://cdut-mars.top/"')
    expect(html).toContain('《火星全球滑坡编目及分类初探》')
    expect(html).not.toContain('原始坐标说明待补')
    const image = html.indexOf('id="case-photo-title"')
    expect(image).toBeGreaterThan(-1)
    expect(image).toBeLessThan(html.indexOf('id="record-data-title"'))
  })
  it('shows hemispheres and keeps the ordinary catalog detail data-only', async () => {
    const html = await render('/catalog/ls_0493')
    expect(html).toContain('西经')
    expect(html).toContain('南纬')
    expect(html).not.toContain('id="case-photo-title"')
  })
  it('states that a case without a public image has no image', async () => {
    const html = await render('/atlas/case/case-ls-0591')
    expect(html).toContain('暂无公开影像')
    expect(html).not.toContain('<img')
  })
  it('marks the cases without a public image on the type page', async () => {
    const html = await render('/atlas/type/SL')
    expect(html.match(/暂无公开影像/g)?.length).toBe(1)
  })
  it('rejects an unknown type politely', async () => {
    const html = await render('/atlas/type/XX')
    expect(html).toContain('未知的滑坡类别')
  })
})
