# 目录导出（第 2 周）

脚本：`scripts/export_checked_catalog.py`。它复用后台已有的 `scripts/export_catalog.py`
（记录级契约与字段校验、原子写入）和 `api/sqlite_store.build_database`（建库），
自己只加四件事：准入闸门、清单（manifest）、staging/ready 的显式区分、纬度参数透传。

## 1. 两种模式

| | `--mode staging`（默认） | `--mode ready` |
|---|---|---|
| 记录正文 | 原始字段直通，`name` 保留空串 | `build_catalog` 校验并回填 `name`、`summary` |
| 坐标系 | 经度 `east-positive, -180..180`，纬度 `unverified`（原样保留） | 纬度必须为 `planetocentric, degrees`；换算与否由 `--latitude-convention` 决定 |
| 数据库 `release_state` | `staging` | `ready` |
| 前置条件 | 准入无 `fail` | 准入无 `fail` **且** 提供 `--coordinate-evidence` **且** 约定不是 `unverified` |
| 可被 APP 读取 | 否（`load_sqlite_catalog` 会抛错） | 是 |

不带 `--coordinate-evidence` 直接跑 `--mode ready` 会以退出码 1 拒绝，并提示
「ready 导出需要 --coordinate-evidence」；带了依据但仍是 `--latitude-convention unverified` 也会被拒绝，
提示「约定未核实时只能出 staging」。

## 2. 纬度参数（默认不换算）

纬度约定目前**未核准**：论文没写、`.prj` 误标、面积用球体算也不反推输入约定。因此默认
`--latitude-convention unverified`——**数值原样保留、不做换算**，导出只能出 `staging`。

三个脚本（准入、导出、案例映射）共用这一个参数，报告与写库口径一致：

| 取值 | 行为 | 能否 ready |
|---|---|---|
| `unverified`（默认） | 原样保留，`coordinateSystem.latitude = unverified` | 否 |
| `planetocentric` | 原样保留（源清单本身就是行星心纬度） | 是（需依据） |
| `planetographic` | 按 IAU 2000 椭球换算为行星心纬度 | 是（需依据） |

换算实现：`scripts/mars_latitude.py`，`lat_c = atan((b/a)² · tan(lat_g))`，`(b/a)² = 0.98826361`；
影响 2971 条全部位移、最大 0.3382°（约 20 km）、平均 0.10°，位移量写进 `manifest.json` 的
`latitudeConversion`（`from`/`to`/`applied`/`recordsShifted`/`maxShiftDeg`/`maxShiftKm`）。

**为什么默认不换算**：对照 1083 个人工取图框（本地 CTX 投影，行星心），换算会让
736 个原本在框内的点移出框外（越界 3 → 739）；边界多边形越界则从 4 涨到 2271。
`scripts/compare_latitude_conventions.py` 可复现。

> `api.sqlite_store` 的 raw 模式把 `metadata.latitude` 写成 `unverified`，含义是「未核准发布」，
> 与 `--latitude-convention unverified` 是同一层意思。

## 3. 产出

| 文件 | 内容 |
|---|---|
| `catalog.json` | 2971 条记录；`ready` 模式符合 `api/schemas.py` 的 `Catalog`（强校验） |
| `admission.json` / `admission.md` | 本次导出前的准入结果，原样留存 |
| `manifest.json` | 版本、模式、时间、条数、类别分布、有图条数、记录摘要、输入与输出的 SHA-256、坐标声明、纬度参数与换算、准入结论、数据库结果、备注 |
| `*.db`（可选） | 由 `build_database` 生成，只含 `metadata`/`sources`/`landslides`/`assets` |

`manifest.json` 里 `recordsSha256` 是对「按 id 排序后的记录列表」做的规范化摘要，
用于判断两次导出是否等价；`inputs` / `outputs` 记录每个文件的 SHA-256 与字节数，便于老师核对。

## 4. 数据库

- `build_database(..., raw=True)` 写 staging：`metadata.release_state=staging`、纬度 `unverified`、
  `metadata.version=inventory-staging-20261024`、`mode=catalog`。
- `build_database(..., raw=False)` 写 ready：先 `Catalog.model_validate`，再入库。
- 已存在的库不会被覆盖，会抛 `FileExistsError('Database already exists; write a new version')`，
  换一个文件名即可。
- `build_database` 只写目录相关四张表。`cases`、`documents`、`case_documents`、`geometry_links`
  由 `scripts/build_case_mapping.py --sqlite <同一个库>` 写入（见 `cases.md`）。

## 5. 字段口径（与第 1 周数据字典一致）

- `id`：`ls_{source_id:04d}`；`source_id` 与原始 FID 相同。
- `name`：原始全空，ready 模式回填 `滑坡 {source_id}`；staging 模式保留空串。
- `summary`：取原始 `desc`（当前全空）。
- `imageUrl`：留空。预览图通过 `assets` 表指向 `train224/.../slide_XXXXX.png`，避免把 1083 个文件塞进 JSON 引用。
- `source`：`{"title": "老师提供的火星滑坡编目", "url": "https://doi.org/10.13203/j.whugis20250300"}`。
- `isDemo`：恒为 `false`；`mode` 恒为 `catalog`。

## 6. 命令

```bash
# 只做检查
python scripts/admission_report.py --handoff <交接包> --json out/admission.json --md out/admission.md

# staging 导出（含 SQLite），默认口径
python scripts/export_checked_catalog.py --handoff <交接包> --out out/staging --mode staging --sqlite out/staging.db

# ready 导出（含 SQLite），这是 APP 能读的那一份；需依据 + 已核实的约定
python scripts/export_checked_catalog.py --handoff <交接包> --out out/ready --mode ready \
  --latitude-convention planetographic \
  --coordinate-evidence "2026-11-03 老师确认：编目纬度按行星面纬度（planetographic）使用" \
  --sqlite out/ready.db
```

默认 `--version mars-landslides-20261101-v3`。
