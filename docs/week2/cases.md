# 首批 12 例案例资源映射（第 2 周）

脚本：`scripts/build_case_mapping.py`。输入是 `06_templates/cases12.todo.json`（老师建议的 12 例）
与交接包里的目录、几何、裁片索引、CTX 图幅，输出一份 JSON 映射、一份高分辨率申请 CSV
和一张速查表；加 `--sqlite` 时把案例行写进数据库。

## 1. 12 例清单（每类 3 例，全部位于水手谷一带）

| 案例 | 编号 | 类别 | 中心坐标 | 窗口 | 预览 m/px | CTX 宽/px | 图幅 |
|---|---|---|---|---|---|---|---|
| case-ls-0493 | ls_0493 | SL | -82.1148, -4.6952 | 42.75 km | 190.9 | 7126 | E-084_N-08 |
| case-ls-0529 | ls_0529 | SL | -79.2591, -4.8158 | 12.53 km | 55.9 | 2088 | E-080_N-08 |
| case-ls-0591 | ls_0591 | SL | -76.2112, -9.8988 | 9.54 km | 42.6 | 1589 | E-080_N-12 |
| case-ls-0483 | ls_0483 | SP | -83.7147, -4.6219 | 10.95 km | 48.9 | 1826 | E-084_N-08 |
| case-ls-0533 | ls_0533 | SP | -79.1683, -5.0082 | 14.23 km | 63.5 | 2372 | E-080_N-08 |
| case-ls-0594 | ls_0594 | SP | -77.9879, -9.4026 | 16.91 km | 75.5 | 2818 | E-080_N-12 |
| case-ls-0505 | ls_0505 | FL | -81.9571, -4.8645 | 10.23 km | 45.7 | 1706 | E-084_N-08 |
| case-ls-0597 | ls_0597 | FL | -78.2466, -9.3115 | 4.32 km | 19.3 | 720 | E-080_N-12 |
| case-ls-0688 | ls_0688 | FL | -78.4314, -0.6905 | 32.54 km | 145.3 | 5423 | E-080_N-04 |
| case-ls-0482 | ls_0482 | CL | -83.9275, -4.7731 | 8.60 km | 38.4 | 1432 | E-084_N-08 |
| case-ls-0528 | ls_0528 | CL | -79.4363, -4.7954 | 26.74 km | 119.4 | 4457 | E-080_N-08 |
| case-ls-0592 | ls_0592 | CL | -77.4230, -9.2294 | 44.45 km | 198.4 | 7408 | E-080_N-12 |

「窗口」是预览裁片的实际边长（1.5× 手工框长边，居中），不是固定 8.6 km；米/像素随目标大小变化，
每张图都不一样。「CTX 宽/px」是按 6 m/像素铺满该窗口需要的像素数，超过 4000 就要拼相邻图幅。

表里的中心坐标是**行星心纬度**（已换算）；源清单与取图框都是行星面纬度，原值在
`cases12.mapping.json` 的 `geometry.point.latitudeSource` 里，换算规则见 `export.md` 第 2 节。

## 2. 每个案例映射了什么

`cases12.mapping.json` 的 `cases[]`：

| 字段 | 含义 |
|---|---|
| `id` / `landslideId` / `sourceId` | `case-ls-0493`、`ls_0493`、`493` 三者互相对应 |
| `title` | 占位标题「{类别中文}案例 · ls_XXXX」，第 3 周改写 |
| `description` | 空串，第 3 周撰写 |
| `status` | `draft`，专家确认后才改 `ready` |
| `imageAssetId` | `train224-493`，指向数据库里已存在的裁片资产 |
| `preview` | 裁片相对路径、SHA-256、尺寸、窗口米数、米/像素、所用 CTX 图幅 |
| `geometry.point` | 点位资产 id、shapefile 第 0 基记录号、原始类型、经纬度 |
| `geometry.polygon` | 边界资产 id、记录号、原始类型、球面面积、顶点数 |
| `citation[]` | 文献 id + 定位（如「PDF 第 8 页 3.2.1 节；表 2（第 8—9 页）；图 4（第 10 页）」） |
| `highResolution` | 取图框（经纬度边界）、窗口米数、CTX 与 HiRISE 像素需求、申请状态 `pending_request` |
| `needsExpertConfirmation` | 恒为 `true`：类别来自老师的编目，本包没有独立判别 |

## 3. 生成时做的校验

每条案例都会核对：目录记录存在、裁片文件存在且 SHA-256 与索引一致、裁片记录的图幅名在 CTX 网格里、
案例中心落在该图幅范围内、边界球面面积与目录 `area` 相对偏差 ≤0.5%。任一条不过就进 `issues`
并以退出码 1 结束。本周 12 例全部通过（0 issues）。

排查过程中修正过两个口径问题，README 里没有写、容易再踩：

1. `02_vectors/ctx_grid` 是一个 shapefile，图幅名在 `Name` 字段里，文件名本身是乱码（GBK 字节被当成 cp437），
   不能拿文件名当图幅名。
2. `image_index.json` 的 `native_window_world_m` 是屏幕坐标：`x = lon·R`，但 `y` 向南为正，所以
   `lat = −y·(180/πR)`。全量 1083 张验证：按此换算中位偏差 0.002 度，去掉负号则中位偏差 15.2 度。
   这条 0.002 度的中位偏差同时说明**取图框与目录用的是同一套纬度约定（行星面）**，因此案例里的
   取图框也随目录一起换算成行星心纬度；不换算的话 40 km 窗口里点位会偏出画面。

## 4. 高分辨率申请

`cases12.highres.csv`（UTF-8 带 BOM，Excel 直接打开）一行一个案例，列包括
`case_id, source_id, landslide_id, type, title, center_lon, center_lat, lon_min, lon_max, lat_min, lat_max,
window_m, preview_m_per_px, ctx_width_px_6m, hirise_width_px_0_3m, requested_display_px, tile, request_status, note`。

给老师的话术：按框内范围出 1024 像素宽的展示图即可，源数据用 CTX（6 m/像素）足够；
HiRISE 0.3 m/像素在这个窗口下宽度会到 1.4–14.8 万像素，只有小案例（窗口 <5 km）才现实。
申请状态现在是 `pending_request`，拿到图后改为 `provided` 并在 `preview` 里补新资产。

## 5. 写入数据库

```bash
python scripts/build_case_mapping.py --handoff <交接包> --out out --sqlite out/ready.db
```

要写进 **ready 库**（APP 能读的那份）。`build_database` 一次只能建新库，所以流程是：先
`export_checked_catalog.py --mode ready --sqlite ready.db` 建库，再把 `ready.db` 复制成 `cases.db`
写案例，或者直接对 `ready.db` 追加（`INSERT OR REPLACE`，可重复执行）。
会插入 `cases` 12 行、`documents` 1 行（论文）、`case_documents` 12 行（每例一条定位）、
`geometry_links` 24 行（点位 + 边界），并顺带建 24 条 `assets`（`geometry-point-XXXX`、
`geometry-polygon-XXXX`，`mime_type=application/x-shapefile`，`metadata_json` 里带记录号与原始类型）。
图片资产沿用 `build_database` 已经建好的 `train224-XXXX`，不重复插入。全部用 `INSERT OR REPLACE`，可重复执行。

## 6. 公开范围

仓库里只放脚本和文档，不放案例数据文件：`scripts/check_public_assets.py` 规定被跟踪的 `.json`
只能包含 `PUBLIC_SOURCE_IDS = {482,483,493,505,528,529,533,597}` 这 8 条（每类 2 条）的记录。
12 例的完整映射（含 591、594、688 等未公开编号）只出现在本地交付包 `D:\python_project\data\第2周_2026-10-26_11-01`
与数据库里，不进 git。第 3 周若要在 APP 里展示 12 例，需要先请老师确认这 12 条的公开范围。

## 7. 留给第 3 周

- 12 条 `description` 与展示用图片（1024 像素）
- 老师对「典型案例」的确认：现在只是按类别各取 3 条，没有主观筛选依据
- FL 只有 15 张裁片，`ls_0688` 是其中面积最大的，若老师认为不够典型，需要换一条并重新申请取图
