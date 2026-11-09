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
| | `coords.latitude_convention` | 记录源纬度约定（默认 unverified，原样保留）；显式声明 planetographic 时才报告换算位移量，并附人工框越界对照 |
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

结论 `conditional_pending_coordinate_verification`（纬度约定尚无书面依据，导出停在 staging），`blockers` 为空。四项警告：

1. `coords.dms`（93 条）：纬度为 0–1 度之间时，导出的度分秒文本丢了负号
   （例 `source_id 686`：数值 `-0.425023`，文本 `0°25′30.08″`）。数值字段正确，导入时以数值为准。
2. `coords.point_in_polygon`（4 条，`source_id` 382、585、1082、1242）：点位在自各边界外
   0.003–0.008 度（≤0.5 km）。这些点不是顶点均值，也不是质心，属于老师标注习惯带来的偏差；
   按 0.5% 的 warn 阈值处理，不阻断导出。
3. `missing.text`（2971+2971）：`name`、`desc` 全空。按契约回填 `滑坡 {source_id}`，说明留空，
   等第 3 周写案例文案。
4. `missing.images`（1888 条缺）：2971 条里 1083 条有裁片（SL 531/1787、SP 94/321、FL 15/50、
   CL 443/813）。只有这 1083 条能直接进图鉴，其余需要另外取图。

## 3. 坐标约定：经度已核，纬度未核

- 经度已核实：东经为正、−180..180，2444 条负值，与论文图 2 的全球分布一致。
- **纬度：未核准。** 论文正文没写、`.prj` 误标 `GCS_WGS_1984`、面积用球体算也不反推输入约定，
  目前没有任何一手材料能证明是行星面还是行星心纬度。
- 因此默认 `--latitude-convention unverified`：**保留原坐标、不做换算**，准入结论只能是
  `conditional_pending_coordinate_verification`，导出只能出 `staging`。

为什么不「先按行星面换算」：对照 1083 个人工取图框（框在本地 CTX 配套投影下，行星心）——

| 判据 | 原值 | 换算后 |
|---|---|---|
| 落在所属边界多边形内 | 越界 4 / 2971 | 越界 2271 / 2971 |
| 落在人工取图框内 | 越界 3 / 1083 | 越界 739 / 1083（**736 个原本在框内的点被移出**） |

`scripts/compare_latitude_conventions.py` 可复现这张表。等拿到可追溯的书面依据后：

- `--latitude-convention planetographic` 会按 IAU 2000 椭球 `(b/a)²` 换算（`lat_c = atan((b/a)²·tan(lat_g))`），
  2971 条全部位移、最大 0.3382°（约 20 km），位移量写进 `manifest.json` 的 `latitudeConversion`；
- `--latitude-convention planetocentric` 表示源清单本身就是行星心纬度，数值原样发布；
- 两种情况都必须同时给 `--coordinate-evidence`，`ready` 才会放行。

准入脚本用与外导**同一个** `latitude_convention` 参数，报告里的 `latitude_convention`、
`latitude`、`manual_boxes` 三处与分析结果一致，不会出现「报告按 A 算、写库按 B 算」。

## 4. 复现

```bash
# 默认口径：纬度原样保留，只出 staging
python scripts/admission_report.py --handoff <交接包> --json admission.json --md admission.md
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode staging --sqlite staging.db

# 拿到书面依据后才能出 ready（二选一：planetocentric 表示源本身就是行星心）
python scripts/export_checked_catalog.py --handoff <交接包> --out <目录> --mode ready \
  --latitude-convention planetographic --coordinate-evidence "<老师书面确认待补：纬度约定与依据；未确认前不要填这一项>" \
  --sqlite ready.db
```

`api/sqlite_store.py` 的读取函数仍会拒绝 `release_state=staging` 的库，`ready` 库才能被 APP 读取。
