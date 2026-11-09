# 结构化关联数据与可检索文本（交给 Agent）

## 1. 为什么由前端内容文件导出

图鉴文案、案例观察要点和引用位置都在 `mobile/src/content/atlas.json`；Agent 需要的是同一份内容的
可检索形态。`scripts/export_atlas_content.py` 从这一处源文件生成四份交付物，避免“页面文案”和
“知识库文本”各写一份、改一处忘一处。

```bash
python scripts/export_atlas_content.py \
  --out <交付目录> \
  --catalog <私有目录>/01_catalog/mars_landslides.json   # 可选，带上就能关联坐标/面积
```

## 2. 产物

| 文件 | 形态 | 用途 |
| --- | --- | --- |
| `atlas-content.json` | JSON | 与 `atlas.json` 同构，`cases[].record` 内嵌关联到的目录记录（含 lon/lat/area） |
| `atlas-cases.csv` | CSV（UTF-8 BOM） | 案例↔目录关联表：case_id / landslide_id / type / 面积 / 经纬度 / 图幅 / 尺度 / in_catalog |
| `atlas-citations.csv` | CSV（UTF-8 BOM） | 引用登记：owner_kind(type/comparison/case) / owner_id / document_id / locator |
| `atlas-search.jsonl` | JSONL | 18 条检索片段（4 类 + 12 案例 + 2 对比），`{id, kind, title, text}` |
| `atlas-export-summary.json` | JSON | 计数：案例数、可关联数、引用数、片段数、按类分布 |
| `atlas-limits.md` | Markdown | 6 条判读限制，可直接作为回答里的“边界说明” |

## 3. 检索片段的约定

每行一个 JSON 对象：

```json
{"id": "case-ls-0493", "kind": "case", "title": "远程堆积的滑移型",
 "text": "案例 case-ls-0493（SL 滑移型）远程堆积的滑移型\n副标题：12 例中最大的 SL\n水手号峡谷 · E-084_N-08 · 190.9 m/像素\n观察要点：…\n标签：大尺度；远程堆积\n引用：3.2.1 节（PDF 第 8 页）；表 2（第 8—9 页）"}
```

- `id` 前缀即类型：`type-SL`、`case-ls-0493`、`compare-sl-vs-sp`；与页面路由一一对应，回答里可以带链接。
- `text` 不含经纬度：位置类问题由 `atlas-cases.csv` / `atlas-content.json` 的 `record` 字段提供，
  检索文本只负责“定义、特征、怎么认、易混、出处”。
- 同一段文字与页面上显示的完全一致，避免 Agent 复述时与 UI 打架。

## 4. 建议的接入方式（后端）

1. 启动时读 `atlas-search.jsonl`，按 `kind` 建三组片段；`text` 直接切块入向量库，`id` 作为引用锚点。
2. 回答“某类滑坡是什么/怎么区分”时取 `type-*`；问“有没有例子”时取同类 `case-*`；
   问“两者区别”时取 `compare-*` 并附 `atlas-limits.md` 的第 1、4 条。
3. 涉及坐标、面积、影像的问题必须回目录接口（`/api/landslides/{id}`），不要用图鉴文本凑数。
4. 如果后端新增 `/api/atlas`，返回 `atlas.json` 同构对象即可；前端 `parseAtlasContent()` 会做边界校验。

## 5. 验收对照

| 本周验收 | 对应实现 |
| --- | --- |
| 地图查记录 | 探索页 `?focus=ls_xxxx` + 案例页“定位到火星球” |
| 图鉴看案例 | `/atlas/type/:type` 三类案例入口 → `/atlas/case/:id` |
| 两类对比 | `/atlas/compare` 两组表 |
| 收藏回访 | 案例页收藏按钮写 `mars-atlas:saved:v1`，`#/mine` 可见 |
| 对话定位可用 | `atlas-search.jsonl` 的 `id`/`title` 可回链到对应页面 |
| 四类各 ≥2 真实案例 | 3 例/类，共 12 例（见 `cases.md`） |
