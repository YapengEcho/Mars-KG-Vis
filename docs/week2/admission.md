# 目录准入检查（第 2 周）

脚本：`scripts/admission_report.py`。输入是老师的交接包，输出 `admission.json` 与 `admission.md`，
退出码在出现 `fail` 时为 1，便于 CI 或本地脚本串联。

## 1. 检查项与阈值

| 组 | 检查 | 判定口径 |
|---|---|---|
| 编号 | `ids.unique` | `id` 无重复 |
| | `ids.pattern` | 每条 `id` 等于 `ls_{source_id:04d}` |
| | `ids.contiguous` | `source_id` 覆盖 1..2971，无空洞 |
| 类别 | `types.enum` | 只允许 `SL`/`SP`/`FL`/`CL` |
| | `types.counts_vs_paper` | 与论文第 7 页的 1787/321/50/813 完全一致 |
| | `types.alias_mapping` | 原始边界类型 `RL→SL`、`DF→FL` 的别名命中数与目录一致 |
| | `types.point_agreement` | 点位属性表 `类型` 与目录 `type` 逐条一致 |
| 坐标 | `coords.range` | 经度 −180..180，纬度 −90..90 |
| | `coords.duplicates` | 无重复经纬度对 |
| | `coords.latitude_convention` | 记录源纬度约定（行星面）与换算后的位移量（条数、最大角度/公里） |
| | `coords.dms` | 度分秒文本与数值字段一致（容差 0.0000001 度） |
| | `coords.point_vs_catalog` | 点位几何与目录 `lon`/`lat` 一致（容差 0.000005 度 ≈ 0.3 m） |
| | `coords.point_in_polygon` | 点位落在自己的边界内；越界按比例判 warn（≤0.5% 记录）或 fail |
| | `coords.area_check` | 目录 `area` 与边界球面面积一致（相对偏差 ≤0.5%） |
| 缺失 | `missing.text` | `name`、`desc` 空值计数 |
| | `missing.numeric` | `area`、`perimeter` 空值计数 |
| | `missing.images` | 有裁片的记录占比，按类别拆分 |
| 图片 | `images.files` | 裁片文件存在 |
| | `images.hash` | 裁片 SHA-256 与索引一致（`--skip-sha` 可跳过） |
| | `images.scale` | 裁片窗口/目标框长边 = 1.5×，窗口与米/像素落在观测区间 |
| 几何 | `geometry.counts` | 点位、边界、目录三者条数一致 |
| | `geometry.index` | 点位与边界下标覆盖 0..n−1 |

面积公式与 ArcGIS 一致：`A = R²/2 · |Σ(λ₂−λ₁)(sinφ₁+sinφ₂)|`，`R = 3396190 m`。
用平面高斯面积会得到约 `1/cosφ` 的系统偏差（中位 1.0558%），不能用作准入依据。

## 2. 本周结果：18 项 pass，4 项 warn，0 项 fail

结论 `ready_for_import`（`--coordinate-evidence` 已记录老师对纬度约定的确认），`blockers` 为空。四项警告：

1. `coords.dms`（93 条）：纬度为 0–1 度之间时，导出的度分秒文本丢了负号
   （例 `source_id 686`：数值 `-0.425023`，文本 `0°25′30.08″`）。数值字段正确，导入时以数值为准。
2. `coords.point_in_polygon`（4 条，`source_id` 382、585、1082、1242）：点位在自各边界外
   0.003–0.008 度（≤0.5 km）。这些点不是顶点均值，也不是质心，属于老师标注习惯带来的偏差；
   按 0.5% 的 warn 阈值处理，不阻断导出。
3. `missing.text`（2971+2971）：`name`、`desc` 全空。按契约回填 `滑坡 {source_id}`，说明留空，
   等第 3 周写案例文案。
4. `missing.images`（1888 条缺）：2971 条里 1083 条有裁片（SL 531/1787、SP 94/321、FL 15/50、
   CL 443/813）。只有这 1083 条能直接进图鉴，其余需要另外取图。

## 3. 坐标约定：已核准为行星面纬度，导出时换算

- 经度已核实：东经为正、−180..180，2444 条负值，与论文图 2 的全球分布一致。
- **纬度：2026-10-31 老师确认按行星面纬度（planetographic）使用**，依据原样写入
  `admission.json` 的 `coordinate_evidence` 与导出清单的 `coordinateEvidence`。
- 此前只有间接线索（面积/周长由 ArcGIS 用 `GCS_Mars` 球体算出、原始 `.prj` 误标 `GCS_WGS_1984`），
  所以第 1 周把它记成「待核准」；现在这条待办可以关掉，但**换算要做，不能只改标签**。

`api/schemas.CoordinateSystem` 只接受 `planetocentric, degrees`，公开底图也是行星心纬度，
因此 `scripts/export_checked_catalog.py` 在导出边界统一换算：

| 项 | 值 |
|---|---|
| 公式 | `lat_c = atan((b/a)² · tan(lat_g))`，IAU 2000 火星椭球 b/a = 3376200/3396190 |
| 全表影响 | 2971 条全部位移，最大 0.3382°（约 20 km），平均 0.10° |
| 记录位置 | `manifest.json` 的 `latitudeConversion`；案例里保留原始值 `latitudeSource` |
| 未换算的代价 | 点位整体偏南/北最多 20 km，在 40 km 宽的裁片里直接跑出画面 |

准入脚本把这个换算计入 `coords.latitude_convention` 一项，所以报告里能直接看到位移量；
`staging` 与 `ready` 两种导出都换算，避免两个库的点位对不上。

## 4. 复现

```bash
python scripts/admission_report.py --handoff <交接包> --json admission.json --md admission.md \
  --coordinate-evidence "2026-10-31 老师确认：编目纬度按行星面纬度（planetographic）使用"
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode ready \
  --coordinate-evidence "同上" --sqlite ready.db
```

`api/sqlite_store.py` 的读取函数仍会拒绝 `release_state=staging` 的库，`ready` 库才能被 APP 读取。
