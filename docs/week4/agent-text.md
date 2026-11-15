# 供助手使用的文本与来源索引

导出命令：

```bash
python scripts/export_atlas_content.py --out output/week4_atlas_handoff --samples examples/recognition.samples.json
```

`output/` 已在 `.gitignore` 里，这六个文件不提交，只随交接包或周交付目录传递。

## 1. 六个文件

| 文件 | 内容 | 谁用 |
| --- | --- | --- |
| `atlas-content.json` | 内容原文 + 可选 `cases[].record`（带 `--catalog` 时的目录联表） | 崔亚鹏入库、周璇取字段 |
| `atlas-cases.csv` | 12 行案例：编号、类型、标题、尺度、窗口、`public_image` 等 | 人工核对、表格工具 |
| `atlas-citations.csv` | 31 行引用：`owner_kind`、`owner_id`、`document_id`、`locator` | 来源索引 |
| `atlas-search.jsonl` | 18 行检索块，一行一个 JSON | 周璇的检索/对话 |
| `atlas-export-summary.json` | 版本、计数、公开影像数 | 版本核对 |
| `atlas-limits.md` | 7 条判读边界 | 对话兜底、免责说明 |

## 2. 检索块格式

每行一个对象，只有四个键：

```json
{"id": "case-ls-0493", "kind": "case", "title": "远程堆积的滑移型", "text": "案例 case-ls-0493（SL 滑移型）…\n公开影像：有（ls_0493.png，224×224）\n引用：…"}
```

| `kind` | 条数 | `id` 形如 | 用途 |
| --- | --- | --- | --- |
| `type_intro` | 4 | `type-SL` | 四类定义、可观察特征、易混类型 |
| `comparison` | 2 | `compare-sl-vs-sp` | 两组对比 |
| `case` | 12 | `case-ls-XXXX` | 单个案例的观察要点 |

合计 18 块。`text` 里不带经纬度、不带面积，也不带影像地址；需要这些值时按 `case` 块的 `id`
去目录接口取，避免助手说出与目录不一致的数字。

## 3. 与 `api/agent/retrieval.py` 的差异

`retrieval.py` 现在吃的是 `{"docs": [...]}`，每项 `doc_id`、`doc_type`、`title`、`text`、`slide_ids`、
`source_title`、`source_url`。本周仍按交接文档的 JSONL 输出，由周璇做一层映射：

| JSONL | retrieval 文档 |
| --- | --- |
| `kind: "type_intro"` | `doc_type: "type_intro"` |
| `kind: "comparison"` | `doc_type: "compare"` |
| `kind: "case"`（新增类型） | `doc_type: "case"`，`slide_ids: ["ls_XXXX"]` |
| `id` | `doc_id` |
| `title` / `text` | 同名 |
| — | `source_title`/`source_url` 取 `sourceDocument.title/url` |

映射只做字段改名，不改文本内容；`case` 类型是新增的，检索侧需要接受它。

## 4. 来源索引

- 唯一文献：`doc-inventory-paper`《火星全球滑坡编目及分类初探》，
  `https://doi.org/10.13203/j.whugis20250300`，页码为 PDF 物理页。
- `atlas-citations.csv` 给出每条引用挂在谁身上（`type` / `comparison` / `case`）、页码定位；
  页面上的“引用与出处”区块与这份 CSV 同源。
- 案例的图片与尺度来源不是论文，而是交接包的 224 像素裁片：图幅名、窗口边长、m/像素 在
  `atlas-cases.csv`（交付包内）与 `docs/week4/atlas-asset-check.md` 里；检索文本只写“公开影像：有/暂无”，不写路径。

## 5. 入库

```bash
python scripts/import_atlas_sqlite.py --database <完整目录库> --input mobile/src/content/atlas.json
```

导入要求 `landslides` 表里已有对应 `landslideId` 且类型一致；8 条公开记录的库放不下 12 例，
用完整目录库试导入过：4 类、12 例、2 组对比，草稿状态，未 `--publish`。`--publish` 只表示文本已复核，
不代表坐标约定已核准。
