# 第 2 周：目录校验、导出脚本与首批案例映射

本周（2026-10-26 — 11-01）负责人的任务是「校验目录编号、类别、坐标和缺失值，交导出脚本与准入报告，
完成首批案例资源映射」。仓库里落地的是四个脚本加这一组说明文档；交付包在
`D:\python_project\data\第2周_2026-10-26_11-01`。

## 1. 脚本

| 脚本 | 作用 | 产出 |
|---|---|---|
| `scripts/admission_report.py` | 目录准入检查：编号、类别、坐标、纬度换算、缺失值、几何、图片七组共 22 项 | `admission.json`、`admission.md` |
| `scripts/mars_latitude.py` | 行星面纬度 ↔ 行星心纬度换算（IAU 2000 火星椭球），单一实现点 | 被下面两个脚本引用 |
| `scripts/export_checked_catalog.py` | 先跑准入，再按 `api/schemas.py` 契约导出目录；staging / ready 两种模式；两种模式都换算纬度 | `catalog.json`、`manifest.json`，可选 SQLite |
| `scripts/build_case_mapping.py` | 把 12 个候选案例连到目录记录、裁片、点位、边界、文献定位与高分辨率申请框 | `cases12.mapping.json`、`cases12.highres.csv`、`cases12.table.md` |

调用顺序固定为「准入 → 导出 → 案例」，后一步都直接复用前一步的函数，不重复实现校验规则：

```bash
python scripts/admission_report.py --handoff <交接包> --json <目录>/admission.json --md <目录>/admission.md \
    --coordinate-evidence "2026-10-31 老师确认：编目纬度按行星面纬度（planetographic）使用"
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode ready \
    --coordinate-evidence "同上" --sqlite <目录>/ready.db
python scripts/build_case_mapping.py --handoff <交接包> --out <目录> --cases <交接包>/06_templates/cases12.todo.json \
    --sqlite <目录>/cases.db
```

## 2. 本周结论

- 编号、类别、坐标数值、几何、图片等 18 项通过，0 项失败。
- 4 项警告都属于「源数据本身的已知情况」，不是本周引入的错误，见 `admission.md`。
- **纬度约定已核准（2026-10-31 老师：行星面纬度 planetographic）**，准入结论因此是
  `ready_for_import`，可以出 `ready` 目录。导出时按 IAU 2000 椭球 `(b/a)²` 换算为行星心纬度，
  2971 条全部位移，最大 0.3382°（约 20 km）。没有这一步就入库，点位会整体偏南/北最多 20 km。
- 12 例全部映射成功，0 个缺项：裁片、点位、边界、CTX 图幅、面积比对、文献定位都能对上；
  案例里的经纬度与取图框同样已换算为行星心纬度，原始行星面纬度留在 `latitudeSource`。

## 3. 文档

- `admission.md`：检查项、阈值、4 项警告的成因与处理口径
- `export.md`：staging/ready 的差别、纬度换算、manifest 字段、数据库行为
- `cases.md`：12 例清单、资源映射字段、高分辨率申请流程、公开范围
- `run.md`：环境准备、命令、测试、已知坑

## 4. 未做（按分工不属于本周，或需要他人输入）

- 12 例的公开范围（仓库白名单只放行 8 条）与「是否典型」仍待老师确认
- 案例正文文案与 1024 像素展示图（第 3 周，且需要外网取图）
- 名称与说明的正式内容（原始 `name`、`desc` 全空，本周只按契约回填占位名）
- SQLite 的完整迁移（本周只写数据库行，迁移与 API 接入由后台负责）
