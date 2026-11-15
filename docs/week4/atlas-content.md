# 图鉴内容：字段口径、引用口径与版本改动

内容文件只有一个：`mobile/src/content/atlas.json`。页面（`AtlasTypeView`/`AtlasCompareView`/`AtlasCaseView`）、
导出脚本和入库脚本都读这一份，不另写助手文本。

## 1. 结构

| 字段 | 含义 |
| --- | --- |
| `version` | 内容版本，跨页面/导出/入库共用同一个值 |
| `sourceDocument` | 唯一来源文献：`id`、`title`、`url`（HTTPS）、`note`（页码口径） |
| `dataNote` | 数据边界说明，页面底部与导出文件都带 |
| `types[4]` | 四类说明：`name`、`english`、`quote`、`deposits`、`flow`、`recognize[]`、`confusable`、`citations[]`、`caseIds[]` |
| `comparisons[2]` | 两组对比：`sl-vs-sp`、`sp-vs-cl`，含 `why`、`left`、`right`、`rows[]`、`citations[]` |
| `limits[7]` | 判读边界与使用限制 |
| `cases[12]` | 案例：`id`、`landslideId`、`type`、`title`、`subtitle`、`region`、`tile`、`previewScale`、`observations[3]`、`tags[2]`、`citationLocator` |

内容文件**不写**经度、纬度、面积、影像地址和数字型 `sourceId`：这些值留在目录与数据库里，
页面在运行时用 `landslideId` 去已加载的目录里取（`lib/atlas.ts` 的 `caseRecord()`）。
这样公开仓库里既没有编目数值，图鉴也不会和目录各说一套。

## 2. 引用口径

- 页码一律是 **PDF 物理页码**，不是论文印刷页码；`sourceDocument.note` 里写明这一点。
- 四类的 `citations` 必须包含「表 2（第 8—9 页）」这一条，另加各自小节：3.2.1（第 8 页）、3.2.2（第 10 页）、
  3.2.3（第 11—12 页）、3.2.4（第 12—13 页）；SL 另引图 4（第 10 页）、SP 图 5（第 11 页）、FL 图 6（第 12 页）。
- 案例的 `citationLocator` 写成「小节（PDF 第 X 页）；表 2（第 A—B 页）；图 N（第 Y 页）」的形式，
  与 `docs/week1/12-cases.md` 的写法一致，便于人工对照。
- 引用只引专家编目的定义与类型描述，不从论文里摘具体案例的坐标或面积。

## 3. 案例文字口径

每条案例三段观察要点，按「先看什么 → 这张图能看到什么 → 这张图看不到什么」组织：

1. 第一段指认该类型的关键形态，并说明它与最易混类型的分界。
2. 第二段结合该案例的 `previewScale` 说明分辨率能支持到什么程度（例如 190.9 m/像素下纵向条纹只能看走向）。
3. 第三段给出可比对象或尺度参照，避免把单张图当成类型判据。

案例标题是描述性的（如“远程堆积的滑移型”），不使用“典型”“标准”一类未经核定的说法；
`tags` 只放可核对的属性（尺度、堆积特点、图幅位置）。

## 4. 版本与改动

### `atlas-20261115-v3`（本周）

| 改动 | 位置 | 原因 |
| --- | --- | --- |
| 版本号 `atlas-20261108-v1` → `atlas-20261115-v3` | 顶层 `version` | 交接文档要求“页面、导出、入库同一个版本” |
| 12 个案例标题后统一加编号（如“远程堆积的滑移型 · ls_0493”） | `cases[].title` | 页面、卡片、检索块都能一眼对上目录编号 |
| 缺图提示改为“仓库不收录影像资源” | `AtlasCaseView.vue`、`limits[6]` | 影像留在交接包，仓库只放文本与可核对的计数 |
| 新增第 7 条 `limits`：仓库不收录影像资源，缺影像的案例页显示“暂无公开影像”，助手不得说“可以看这张图” | `limits[6]` | 12 例里 4 例没有公开预览，必须先在文本里挡住错误说法 |
| 案例页无影像时给出明确提示 | `AtlasCaseView.vue` | 之前只是不显示图片，容易被理解成加载失败 |
| 类型页案例卡片给缺图案例加“暂无公开影像”标签 | `AtlasTypeView.vue` | 让图鉴入口就能看出哪几例只有文字 |
| 导出增加 `public_image` 列与“公开影像：有/暂无”检索行 | `scripts/export_atlas_content.py` | 助手回答问题时要能区分“有图”和“只有文字” |
| 新增资源核对脚本 | `scripts/check_content_assets.py` | 逐例核对编号、类型、图幅、尺度、哈希、引用 |

v2 没有改动四类定义、对比表、12 例的 `id`/`landslideId`/`type`/观察要点/引用，改的都是“边界与状态”这一类信息，
所以页面上的文字结论与 v1 一致，只是把“哪几例没有公开影像”说清楚了。

## 5. 校验规则

`scripts/export_atlas_content.py --check` 会拒绝：版本缺失、来源不是 HTTPS、类型不是四类、
`caseIds` 与 `cases[].id` 不等、案例 `id` 不合 `case-ls-XXXX`、`landslideId` 不合 `ls_XXXX`、
对比两侧相同、`recognize` 为空、`limits` 为空。

`api/content_store.py` 在此基础上还要求：`landslideId` 必须在 `landslides` 表里且 `type` 一致、
每条引用 `documentId` 等于 `sourceDocument.id` 且 `locator` 非空、案例 `title` 与 `citationLocator` 非空。
本周用完整目录库试导入，四类、12 例、2 组对比全部通过（草稿状态，未 `--publish`）。
