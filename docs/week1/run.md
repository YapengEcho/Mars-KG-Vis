# 第 1 周脚本运行说明

## 1. 环境

| 项目 | 要求 |
|---|---|
| Python | 3.11 及以上（本次实测 3.14） |
| 第三方库 | `pyshp`（仅几何核对需要，`pip install pyshp`）；其余全部是标准库 |
| 交接包 | 目录结构须包含 `handoff_manifest.json`、`01_catalog/`、`02_vectors/`、`03_images/`、`06_templates/` |
| 编码 | 数据文件按 UTF-8 读取（脚本自动兼容 BOM）；SHP 属性为 UTF-8，`.cpg` 已标明 |

Windows 上如果 `python` 不是 3.11+，用完整路径调用解释器即可。

## 2. 校验交接包

```bash
python scripts/handoff_check.py --handoff D:\python_project\teng_jiajie_week1_20261024
```

可选参数：

| 参数 | 作用 |
|---|---|
| `--json report.json` | 输出结构化结果（给 CI 或后续脚本用） |
| `--md report.md` | 输出 Markdown 表格 |
| `--skip-sha` | 跳过 1083 张图的 SHA-256（约省 10 秒；只在确认文件未动时用） |

输出是一行一项检查：

```
PASS catalog.records: 2971 records (expected 2971)
...
SUMMARY {'pass': 27, 'warn': 5}
```

退出码：`0` 表示没有 `fail`；`1` 表示存在 `fail`，应阻断后续流程。`warn` 不影响退出码，但必须人工确认——目前 5 个 warn 都是“已知且已记录”的：

| warn | 含义 | 处理 |
|---|---|---|
| `catalog.optional_fields` | 2971 条的 `name`/`desc` 全为空 | 按契约回填“滑坡 + 编号” |
| `images.coverage` | 仅 36.5% 的记录有裁片 | 正常，图片不覆盖全量 |
| `sample.coordinate_status` | 样例标记了纬度待核准 | 预期行为 |
| `shapes.dms_vs_geometry` | 93 条度分秒文本丢负号 | 已记录，用数值字段 |
| `coordinate.convention` | 纬度约定未核准 | 待老师确认 |

## 3. 生成 20 条样例与 12 例清单

```bash
python scripts/build_week1_artifacts.py --handoff D:\python_project\teng_jiajie_week1_20261024 --out out\week1
```

输出三个文件：

| 文件 | 内容 |
|---|---|
| `records20.json` | 20 条样例（APP 契约形状 + `handoff` 追溯块） |
| `cases12.json` | 12 个案例候选（含缺项、来源、裁片 SHA-256） |
| `tables.md` | 上面两份的 Markdown 表格，可直接贴进报告 |

注意：这三份产物**不要提交到仓库**。`scripts/check_public_assets.py` 只允许公开 8 条真实记录，20 条样例与 12 例候选超出发布边界；`--out` 指向交付包目录即可。

脚本内有硬校验：样例编号必须存在于目录、坐标与目录一致（容差 5×10⁻⁶ 度）、裁片存在且 SHA-256 匹配；任一不满足直接退出，不会生成“看起来对”的文件。

## 4. 单元测试

```bash
python scripts/test_handoff_check.py
```

9 个用例，覆盖：度分秒解析（含丢负号情形）、行星面/行星心纬度差、报告汇总逻辑。不依赖 pytest；装了 pytest 也可以直接 `pytest scripts/test_handoff_check.py`。

## 5. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `pyshp not installed` | 未装 pyshp | `pip install pyshp`；不装也能跑，只是跳过几何检查并给 warn |
| `does not look like the handoff package` | `--handoff` 指错了层级 | 指向包含 `handoff_manifest.json` 的那一层 |
| SHP 读取出 `UnicodeDecodeError` | 用了别的库默认 GBK | 本脚本固定 `encoding="utf-8"`，不要改 |
| 中文路径报错 | 旧版 Python | 用 3.11+；本次实测中文路径正常 |

## 6. 运行记录（2026-10-24）

| 项目 | 结果 |
|---|---|
| `handoff_check.py` | 27 PASS / 5 WARN / 0 FAIL，退出码 0 |
| `build_week1_artifacts.py` | 20 条样例 + 12 例生成成功，所有裁片 SHA-256 匹配 |
| `test_handoff_check.py` | 9/9 通过 |
| 耗时 | 全量（含 1083 张图校验）约 15 秒 |
