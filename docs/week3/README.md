# 第 3 周（2026-11-02 — 11-08）图鉴与案例页

本周负责人的任务：**完成四类说明、类型对比与 12 个案例页面，核对引用和标签，交结构化关联数据及可检索文本。**
验收口径：地图查记录、图鉴看案例、两类对比、收藏回访、对话定位可用；四类各至少 2 个真实案例。

## 1. 交付物

| 交付 | 位置 | 说明 |
| --- | --- | --- |
| 四类图鉴页 | `mobile/src/views/AtlasTypeView.vue`（路由 `/atlas/type/:type`） | 定义、堆积/流动特征、识别要点、易混类型、引用出处、每类 3 个案例入口 |
| 两类对比 | `mobile/src/views/AtlasCompareView.vue`（`/atlas/compare`） | SL↔SP、SP↔CL 两张逐维度对照表 + 判读限制 |
| 12 个案例页 | `mobile/src/views/AtlasCaseView.vue`（`/atlas/case/:id`） | 观察要点、标签、目录记录（运行时关联）、引用、同类前后切换 |
| 图鉴文案与关联 | `mobile/src/content/atlas.json` | 只含文案、引用位置和记录编号；坐标/面积/影像运行时从目录读取 |
| 纯函数层 | `mobile/src/lib/atlas.ts` | 内容校验、按类取案例、案例↔目录关联、可检索文本 |
| 结构化导出 | `scripts/export_atlas_content.py` | 生成关联表、引用登记、JSONL 检索片段 |
| 文档 | 本目录 | 信息架构、案例表、Agent 接入、运行说明 |

## 2. 两个不能省的约定

1. **公开仓库不放真实记录。** `atlas.json` 里没有经纬度、面积、影像，也没有整型 `sourceId`；
   案例页通过 `landslideId`（如 `ls_0493`）在运行时向目录 join。`scripts/check_public_assets.py`
   因而继续通过，仓库里只有那 8 条已批准的公开样例。
2. **文案只引用论文，不替代判读。** 每个类型和对比都带引用位置（表 2、3.2.x 节、图 3—6），
   案例页的 `citationLocator` 指向同一篇文献。页码是 PDF 物理页，见 `atlas.md` 第 3 节。

## 3. 本周没做完 / 待确认

- 12 例是否“典型”仍需老师确认；目前按 3 类/类、覆盖三张 CTX 图幅、跨 1.85–581 km² 规模挑选。
- 图鉴页的数据来自本地 `atlas.json`，后端还没有 `/api/atlas` 接口（属崔亚鹏本周资源读取范围）；
  接口就绪后把 `atlas.ts` 里的导入换成 fetch 即可，页面不用改。
- 案例页的“定位到火星球”“收藏”跟探索页共用同一份状态，收藏回访已可用；地图联动（梁先楷）未接。

## 4. 验收自测

```bash
cd mobile
npm test          # 28 个用例，含 atlas 内容与 join 的 12 个
npm run build     # vue-tsc + vite build
cd ..
python scripts/export_atlas_content.py --check
```

页面路径：`#/atlas` → 四个类型 → `#/atlas/type/SL` → `#/atlas/case/case-ls-0493` → `#/atlas/compare`。
