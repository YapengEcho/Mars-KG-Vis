# 图鉴信息架构与内容约定

## 1. 路由

| 路径 | 组件 | 作用 |
| --- | --- | --- |
| `#/atlas` | `AtlasView.vue` | 四类入口（每块显示目录条数与案例数）、两类对比入口、12 案例网格、目录检索 |
| `#/atlas/type/:type` | `AtlasTypeView.vue` | 单类说明页，`type` ∈ `SL/SP/FL/CL`，非法值显示提示而不报错 |
| `#/atlas/compare` | `AtlasCompareView.vue` | SL↔SP、SP↔CL 两组对照表 + 6 条判读限制 |
| `#/atlas/case/:id` | `AtlasCaseView.vue` | 单案例页，`id` 形如 `case-ls-0493` |

旧的 `#/atlas` 行为（四块筛选 + 搜索）保留在页面下半部分，筛选按钮改为独立的 `type-filters` 行，
类型块本身变成通往类型页的链接，避免“点一下只筛了个列表”的误导。

## 2. 内容文件 `mobile/src/content/atlas.json`

```
version           atlas-20261108-v1
sourceDocument    {id, title, url, note}          论文 DOI 与页码口径
dataNote          为什么这里没有坐标/面积/影像
types[4]          {type, name, english, quote, deposits, flow, recognize[], confusable, citations[], caseIds[]}
comparisons[2]    {id, title, why, left, right, rows[{dimension,left,right}], citations[]}
limits[6]         判读限制，对比页原样展示
cases[12]         {id, landslideId, type, title, subtitle, region, tile, previewScale, observations[], tags[], citationLocator}
```

约束由 `mobile/src/lib/atlas.ts` 的 `parseAtlasContent()` 在加载时强制：四类齐全且不重复、
每类至少有定义/堆积/流动/易混/识别要点/引用、案例编号唯一且 `ls_\d{4}`、
每个案例必须登记在对应类型的 `caseIds` 里、对比两侧不同类型、至少两组对比。

**刻意不放进来的字段**：`longitude`、`latitude`、`areaKm2`、`imageUrl`、整型 `sourceId`。
原因有两个：公开仓库只允许 8 条已批准的样例记录（`scripts/check_public_assets.py`），
以及记录值应以目录服务为准，页面不该有第二份副本。

## 3. 案例↔目录的运行时关联

```
case.landslideId ──findSlide(records, id)──▶ Slide {id, sourceId, name, type, longitude, latitude, areaKm2, source}
```

- 目录是 demo 或公开子集时，12 条里有 8 条（`ls_0482/0483/0493/0505/0528/0529/0533/0597`）能关联，
  其余 4 条（`ls_0591/0594/0592/0688`）显示“当前目录里没有这条记录”，只展示图鉴文案。
- 关联不到时不隐藏页面、不伪造坐标：案例页给出明确说明，接入完整目录后自动补全。
- “定位到火星球”跳 `#/?focus=ls_0493&visit=<ts>`，与探索页的 `?focus=` 处理一致；“收藏”写 `mars-atlas:saved:v1`，
  `#/mine` 的收藏回访因此对案例页同样生效。

## 4. 引用口径

- 文献：`火星全球滑坡编目及分类初探`，DOI `10.13203/j.whugis20250300`。
- 页码：**PDF 物理页码**（含封面，共 28 页），不是印刷页码；`sourceDocument.note` 已在页面脚注中说明。
- 类型页引用：表 2（第 8—9 页）、3.2.1（第 8 页）、3.2.2（第 10 页）、3.2.3（第 11—12 页）、3.2.4（第 12—13 页）、
  图 3（第 9 页）、图 4（第 10 页）、图 5（第 11 页）、图 6（第 12 页）。
- 引用登记由 `scripts/export_atlas_content.py` 汇总成 `atlas-citations.csv`（owner_kind / owner_id / locator），
  换版式时只需改 `atlas.json` 里的 locator，不必动页面。

## 5. 与后续接口的衔接

后端目前只有 `/api/catalog` 等目录接口。等新增 `/api/atlas` 时，建议返回与 `atlas.json` 同构的对象
（`types/comparisons/cases/limits`），前端把 `import raw from '../content/atlas.json'` 换成一次 fetch 即可；
`parseAtlasContent()` 继续当边界校验，字段不合同样会拒绝渲染。
