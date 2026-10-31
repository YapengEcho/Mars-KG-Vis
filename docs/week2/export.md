# 目录导出（第 2 周）

脚本：`scripts/export_checked_catalog.py`。它复用后台已有的 `scripts/export_catalog.py`
（记录级契约与字段校验、原子写入）和 `api/sqlite_store.build_database`（建库），
自己只加三件事：准入闸门、清单（manifest）、以及 staging/ready 的显式区分。

## 1. 两种模式

| | `--mode staging`（默认） | `--mode ready` |
|---|---|---|
| 记录正文 | 原始字段直通，`name` 保留空串 | `build_catalog` 校验并回填 `name`、`summary` |
| 坐标系 | 经度 `east-positive, -180..180`，纬度 `unverified` | 纬度必须为 `planetocentric, degrees` |
| 数据库 `release_state` | `staging` | `ready` |
| 前置条件 | 准入无 `fail` | 准入无 `fail` **且** 提供 `--coordinate-evidence` |
| 可被 APP 读取 | 否（`load_sqlite_catalog` 会抛错） | 是 |

不带 `--coordinate-evidence` 直接跑 `--mode ready` 会以退出码 1 拒绝，并提示
「ready 导出需要 --coordinate-evidence」。

## 2. 产出

| 文件 | 内容 |
|---|---|
| `catalog.json` | 2971 条记录，符合 `api/schemas.py` 的 `Catalog`（ready 模式会强校验） |
| `admission.json` / `admission.md` | 本次导出前的准入结果，原样留存 |
| `manifest.json` | 版本、模式、时间、条数、类别分布、有图条数、记录摘要、输入与输出的 SHA-256、坐标声明、准入结论、数据库结果、备注 |
| `*.db`（可选） | 由 `build_database` 生成，只含 `metadata`/`sources`/`landslides`/`assets` |

`manifest.json` 里 `recordsSha256` 是对「按 id 排序后的记录列表」做的规范化摘要，
用于判断两次导出是否等价；`inputs` / `outputs` 记录每个文件的 SHA-256 与字节数，便于老师核对。

## 3. 数据库

- `build_database(..., raw=True)` 写 staging：`metadata.release_state=staging`、纬度 `unverified`、
  `metadata.version=inventory-staging-20261024`、`mode=catalog`。
- `build_database(..., raw=False)` 写 ready：先 `Catalog.model_validate`，再入库。
- 已存在的库不会被覆盖，会抛 `FileExistsError('Database already exists; write a new version')`，
  换一个文件名即可。
- `build_database` 只写目录相关四张表。`cases`、`documents`、`case_documents`、`geometry_links`
  由 `scripts/build_case_mapping.py --sqlite <同一个库>` 写入（见 `cases.md`）。

## 4. 字段口径（与第 1 周数据字典一致）

- `id`：`ls_{source_id:04d}`；`source_id` 与原始 FID 相同。
- `name`：原始全空，ready 模式回填 `滑坡 {source_id}`；staging 模式保留空串。
- `summary`：取原始 `desc`（当前全空）。
- `imageUrl`：留空。预览图通过 `assets` 表指向 `train224/.../slide_XXXXX.png`，避免把 1083 个文件塞进 JSON 引用。
- `source`：`{"title": "老师提供的火星滑坡编目", "url": "https://doi.org/10.13203/j.whugis20250300"}`。
- `isDemo`：恒为 `false`；`mode` 恒为 `catalog`。

## 5. 命令

```bash
# 只做检查
python scripts/admission_report.py --handoff <交接包> --out out --json out/admission.json --md out/admission.md

# staging 导出（含 SQLite）
python scripts/export_checked_catalog.py --handoff <交接包> --out out --mode staging --sqlite out/staging.db

# 拿到坐标依据后升级
python scripts/export_checked_catalog.py --handoff <交接包> --out out --mode ready \
  --coordinate-evidence "依据文字" --sqlite out/ready.db
```

默认 `--version mars-landslides-20261031-v1`，坐标依据不足时不要改版本号，避免和正式版混淆。
