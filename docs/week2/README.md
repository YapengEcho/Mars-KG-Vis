# 第 2 周：目录校验、导出脚本与首批案例映射

本周（2026-10-26 — 11-01）负责人的任务是「校验目录编号、类别、坐标和缺失值，交导出脚本与准入报告，
完成首批案例资源映射」。仓库里落地的是五个脚本加这一组说明文档；交付包在
`D:\python_project\data\第2周_2026-10-26_11-01`。

## 1. 脚本

| 脚本 | 作用 | 产出 |
|---|---|---|
| `scripts/admission_report.py` | 目录准入检查：编号、类别、坐标、纬度约定、缺失值、几何、图片七组共 22 项 | `admission.json`、`admission.md` |
| `scripts/mars_latitude.py` | 纬度约定的单一实现点：`unverified` 原样保留，`planetographic` 时才按 IAU 2000 火星椭球换算 | 被其余三个脚本引用 |
| `scripts/export_checked_catalog.py` | 先跑准入，再按 `api/schemas.py` 契约导出目录；staging / ready 两种模式 | `catalog.json`、`manifest.json`，可选 SQLite |
| `scripts/build_case_mapping.py` | 把 12 个候选案例连到目录记录、裁片、点位、边界、文献定位与高分辨率申请框；写库带外键检查 | `cases12.mapping.json`、`cases12.highres.csv`、`cases12.table.md` |
| `scripts/compare_latitude_conventions.py` | 对比原值与换算后的越界情况，给出「不默认换算」的量化依据 | 终端输出 |

调用顺序固定为「准入 → 导出 → 案例」，后一步都直接复用前一步的函数，不重复实现校验规则。
三个脚本共用同一个 `--latitude-convention` 参数（默认 `unverified`），报告与写库不会出现两套口径：

```bash
# 默认口径：纬度原样保留，只出 staging
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode staging \
    --sqlite <目录>/staging.db
python scripts/build_case_mapping.py --handoff <交接包> --out <目录> --cases <交接包>/06_templates/cases12.todo.json \
    --sqlite <目录>/cases.db

# 只有拿到可追溯的纬度依据后，才允许 ready
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode ready \
    --latitude-convention planetographic \
    --coordinate-evidence "<老师书面确认待补：纬度约定与依据；未确认前不要填这一项>" \
    --sqlite <目录>/ready.db
```

## 2. 本周结论

- 编号、类别、坐标数值、几何、图片等 18 项通过，0 项失败。
- 4 项警告都属于「源数据本身的已知情况」，不是本周引入的错误，见 `admission.md`。
- **纬度约定未核准**：目前没有一手材料能证明编目用的是行星面还是行星心纬度，因此默认
  `unverified`、数值原样保留，导出停在 `staging`。对照 1083 个人工取图框，直接换算会让
  **736 个原本在框内的点移出框外**（越界 3 → 739），所以现在不做换算；换算能力保留在
  `mars_latitude.py`，等依据到位再打开。
- 12 例全部映射成功，0 个缺项：裁片、点位、边界、CTX 图幅、面积比对、文献定位都能对上；
  案例里的点位纬度与取图框保持各自坐标定义，取图框来自本地 CTX 配套投影（行星心），
  **不跟随编目换算**，原始值留在 `latitudeSource`。

## 3. 文档

- `admission.md`：检查项、阈值、4 项警告的成因与处理口径
- `export.md`：staging/ready 的差别、纬度参数、manifest 字段、数据库行为
- `cases.md`：12 例清单、资源映射字段、高分辨率申请流程、公开范围
- `run.md`：环境准备、命令、测试、已知坑

## 4. 未做（按分工不属于本周，或需要他人输入）

- 纬度约定的书面依据（需要老师或组长确认，确认后才可 `ready`）
- 12 例的公开范围（仓库白名单只放行 8 条）与「是否典型」仍待确认
- 案例正文文案与 1024 像素展示图（第 3 周，且需要外网取图）
- 名称与说明的正式内容（原始 `name`、`desc` 全空，本周只按契约回填占位名）
- SQLite 的完整迁移（本周只写数据库行，迁移与 API 接入由后台负责）
