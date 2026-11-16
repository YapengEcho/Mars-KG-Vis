# 第 4 周图鉴内容：定稿与资源核对

本周按 2026-11-11 的交接文档（`agent-recognition.md`）与更新后的 `分工.md` 收口图鉴内容：
只维护 `mobile/src/content/atlas.json` 这一份文本，补逐案例的资源核对表、内容版本与改动说明，
并产出给助手用的检索文本与来源索引。**先给组内看内容与表，暂不提交。**

## 交付物

| 交付物 | 位置 | 说明 |
| --- | --- | --- |
| 内容版本与改动说明 | [atlas-content.md](atlas-content.md) | v1 → v2 的差异、字段口径、引用口径 |
| 图文对应检查表 / 缺项表 | [assets.md](assets.md) + `atlas-asset-check.md`（`.csv` 见交付包） | 12 例逐条核对，8 例公开、4 例缺影像 |
| 四类说明与对比 | `mobile/src/content/atlas.json` | 定义、可观察特征、易混类型、边界，带论文页码 |
| 案例关联 | 同上 `cases[12]` | `landslideId`/`type`/观察要点/引用与 `caseIds` 一一对应 |
| 供助手使用的文本与来源索引 | [agent-text.md](agent-text.md) + 导出文件 | 18 个检索块，含来源与页码 |
| 页面改动 | `AtlasCaseView.vue`、`AtlasTypeView.vue` | 无公开影像时明确写“暂无公开影像” |

## 本周结论

- 内容版本 **`atlas-20261115-v3`**；四类 × 3 例，共 12 例，2 组对比，31 条引用，18 个检索块。
- 12 例中 **8 例**同时有公开目录记录与 224 像素裁片，**4 例**（`ls_0591`、`ls_0594`、`ls_0688`、`ls_0592`）
  只有文字说明与完整目录记录（那 4 张裁片在交接包里存在，但仓库不收影像，页面上显示“暂无公开影像”）。
- 尺度按每张裁片自己的窗口计算（窗口边长 ÷ 224），19.3–198.4 m/像素，未使用统一比例。
- 坐标保留星面原值，本周不做任何批量换算；页面上“原始坐标说明待补”的提示保持原样。

## 运行

```bash
# 内容自检（字段、引用、caseIds 归组）
python scripts/export_atlas_content.py --check

# 资源核对表（需要交接包；只读）
python scripts/check_content_assets.py --out docs/week4

# 六个交接文件（不提交，输出到被 .gitignore 的 output/）
python scripts/export_atlas_content.py --out output/week4_atlas_handoff --samples examples/recognition.samples.json
```

`--out` 生成的六个文件与交接文档约定一致：`atlas-content.json`、`atlas-cases.csv`、
`atlas-citations.csv`、`atlas-search.jsonl`、`atlas-export-summary.json`、`atlas-limits.md`。

注意：脚本产物 `output/week4_atlas_handoff/atlas-content.json` 与本目录的 `atlas-content.md`（改动说明）**不是同一个东西**，
前者是导出的内容与计数，后者是文字说明。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| `export_atlas_content.py --check` | 图鉴内容校验通过 |
| `check_content_assets.py` | 12 例，公开影像 8 例，公开目录 8 例，问题 0 条 |
| `api/content_store` 导入 | 4 类、12 例、2 组对比，草稿状态导入成功 |
| 前端 `npm test` / `npm run build` | 64 passed（10 files）/ 构建通过 |
| Python 单测 | `test_export_atlas_content` 8、`test_check_content_assets` 6 |
| `api` 测试 | 98 passed |
| 发布边界 `check_public_assets.py` | 通过：333 个跟踪文件，仅 8 条公开样例与 8 张预览 |

## 没做完的

- 4 例的公开影像与公开目录记录：需要崔亚鹏确认公开范围，或按流程申请资源后才能补。
- 原始坐标（星面/星心）仍待确认，本周只记录、不换算。
- 全量 2971 条记录的图鉴化不在本周范围。
