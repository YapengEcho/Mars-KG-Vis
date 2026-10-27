# 全球底图资源核对表

版本：v2（2026-10-24，补充 USGS 产品页的许可与纬度类型）｜对应交付：第 1 周“核对全球底图资源”
｜核对人：滕佳杰｜配合：梁先楷（球体映射）

## 0. 结论（v2 调整）

**首选：NASA MGS MOLA DEM 463 m（USGS 发布，128 ppd，行星心纬度，许可 CC0 公有领域）。**
它是**高程数据**，需要配色彩/晕渲才能当球体贴图；USGS WMS 的 `MOLA_color` 图层就是同一产品的彩色晕渲，可直接取图。

**备选：Viking MDIM 2.1 彩色镶嵌（256 ppd ≈ 231 m/像素，光学影像）**——观感更“像照片”，但该产品页没有明示许可，需要补一次许可核对。

两条都是火星全球覆盖、简单圆柱投影、经度 −180…180、东经为正。

## 1. 候选资源

| # | 产品 | 机构 | 分辨率 | 投影／经度 | 纬度类型 | 许可 | 获取 | 状态 |
|---|---|---|---|---|---|---|---|---|
| A | Mars MGS MOLA DEM 463 m | NASA GSFC / MGS-MOLA，USGS Astrogeology 发布 | 128 ppd，463.08 m/像素，46080×23040，16 bit | Simple Cylindrical，Positive East，−180…180 | **Planetocentric（明示）** | **CC0 公有领域** | 单文件 GeoTIFF 2 GB；1024 预览 jpg | **已核实** |
| B | MDIM 2.1 Colorized Global Mosaic（Viking） | USGS Astrogeology；NASA Ames 上色 | 256 ppd ≈ 231 m/像素 | 简单圆柱；服务范围 −180…360 | 未在该图层声明 | 页面未明示，需补核 | USGS WMS 图层 `MDIM21_color` | 已核实服务，许可待核 |
| C | MOLA Topography + THEMIS IR Day Blend | NASA GSFC + ASU | MOLA 128 ppd + THEMIS 100 m | 简单圆柱；−180…约 360 | 未在该图层声明 | 未明示 | USGS WMS `MOLA_THEMIS_blend` | 图层存在 |
| D | Murray Lab 全球 CTX 镶嵌 | Caltech Murray Lab | 约 5–6 m/像素 | 4°×4° 图幅，命名如 `E-084_N-08` | — | 需按其说明引用 | 论文表 1 给出 `https://murray-lab.caltech.edu/CTX/V01/tiles/`（站点当前网络不可达） | 引用地址已核，页面未核 |

### 1.1 已核实的地址与关键字段

**A. MOLA DEM 463 m（首选）**

- 产品页：`https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m`
- 直链（GeoTIFF，2 GB）：`https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
- 1024 px 预览：产品页 Sample
- 产品页原文要点：
  - `Pixel Resolution (meters/pixel) 463.0835744`、`Scale (pixels/degree) 128`
  - `Map Projection Name: Simple Cylindrical`
  - `Latitude Type: Planetocentric`
  - `Longitude Direction: Positive East`、`Longitude Domain: -180 to 180`
  - `Radius A / Radius C: 3396190`
  - `Access Constraints: CC0 (public domain)`、`Use Constraints: None`
- 这组参数与本项目数据**完全一致**：老师算面积用的正是 R=3396190 的火星球体，裁片窗口坐标也是同一套等距圆柱平面。

**B. MDIM 2.1 彩色（备选，光学影像）**

- WMS 服务：`https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/mars/mars_simp_cyl_quads.map&service=WMS&request=GetCapabilities&version=1.3.0`
- 图层名：`MDIM21_color`；服务摘要写明 256 pixels/degree（231 m/p），Viking 影像配准到 MOLA 控制网
- 服务 `MaxWidth`/`MaxHeight` = 4096，全球 8192 宽要分块拼
- 支持 `EPSG:4326` 与 `IAU_2015:49900`

**C. MOLA + THEMIS 混合**：同 WMS，图层 `MOLA_THEMIS_blend`（THEMIS IR Day 100 m 全球镶嵌 + MOLA 晕渲）

**D. CTX 图幅**：论文表 1（PDF 第 6 页）给的就是 Murray Lab 的 `.../CTX/V01/tiles/`，与交接包 `ctx_grid` 里 3960 个 4° 图幅（`E-180_N84` 这种命名）是同一体系。

## 2. 坐标约定陷阱

| 陷阱 | 具体表现 | 处理 |
|---|---|---|
| 经度 0–360 | ASU 分块按 180–210°E 命名；WMS 服务范围写到 360 | APP 统一 −180…180 东经为正；重采样时以 −180 为左边界 |
| 经度方向 | 首选 A 明确 Positive East；B 需在图例上确认 | 若拿到西经为正的产品，必须取反 |
| 行星心 vs 行星面 | A 明示 Planetocentric；B/C 未声明 | 目录纬度约定仍未核准（见数据字典第 3 节），先不动底图 |
| 假彩色 | MOLA 彩色是高程配色，不是可见光 | 图例必须写“颜色表示高程”，不能当“火星真实颜色” |
| 极区 | MOLA 原始在 88°–90° 有空洞，现版本已用极区成果补齐；数据在 87° 以上仍稀疏 | 极区不放大解释 |

## 3. 球体映射核对（Three.js）

现状（`mobile/src/components/MarsGlobe.vue`）：球体半径 2，纹理是程序生成的 Canvas 示意图，没有加载任何真实底图。

`mobile/src/lib/marsCoordinates.ts` 的 `lonLatToCartesian`：

```
x = R·cosφ·sinλ      y = R·sinφ      z = R·cosφ·cosλ
```

即 **0° 经线朝 +Z，北极朝 +Y**，与 Three.js `SphereGeometry` 的默认 UV 不一致：

| 纹理 u | Three.js `SphereGeometry` 默认落在 | 标准等距圆柱图在该 u 的经度 | 本项目对应经度的方向 |
|---|---|---|---|
| 0.00 | −X | −180° | −Z |
| 0.25 | +Z | −90° | −X |
| 0.50 | +X | 0° | +Z |
| 0.75 | −Z | +90° | +X |

（已用 `x = −R·cos φ·sin θ, z = R·sin φ·sin θ` 逐点验算。）

**结论：直接贴一张标准等距圆柱图（左 −180、中 0、右 +180）会整体旋转 90°。**

两种改法（任选其一，建议第二种）：

1. 纹理偏移：`texture.wrapS = THREE.RepeatWrapping; texture.offset.x = 0.25`。
2. 改球体起始角：`new THREE.SphereGeometry(R, 128, 64, -Math.PI / 2, Math.PI * 2)`，把 u=0 挪到 −Z。

**验证方法**（梁先楷联调时做）：加载底图后对准 0°E、0°N，应与 Mars Trek 显示的同一位置一致；再叠加 `ls_0493`（82.12°W, 4.76°S），应落在 Valles Marineris 西段；`ls_0592`（77.43°W, 9.31°S）应在其东南方向。

## 4. 分辨率与体积建议

| 用途 | 尺寸 | 预计 JPEG | 说明 |
|---|---|---|---|
| 手机预览 | 2048×1024 | 0.3–0.6 MB | 够看峡谷带 |
| 推荐 | 4096×2048 | 0.8–1.8 MB | 手机/网页流畅；WMS 单次可取（≤4096） |
| 桌面高清 | 8192×4096 | 2.5–5 MB | 需 2 次 WMS 拼接，或用 ASU / DEM 重采样 |

不要直接上 46080×23040（约 10.6 亿像素），Three.js 会爆显存。

## 5. 署名文案

首选（MOLA DEM，CC0）：

> 底图：NASA/MGS MOLA 高程数据（463 m/像素，128 ppd），USGS Astrogeology 发布，CC0 公有领域；本图经重采样与色彩处理用于球体显示。

英文：

> Basemap: NASA/MGS Mars Orbiter Laser Altimeter (MOLA) DEM, 463 m/px (128 ppd), published by USGS Astrogeology (CC0 public domain). Resampled and shaded for this viewer.

备选（MDIM 2.1）：

> 底图：NASA/USGS Viking MDIM 2.1 Colorized Global Mosaic（256 ppd，约 231 m/像素），经重采样与坐标重排用于球体显示。

## 6. 落表（对应 `06_templates/basemap.todo.json`）

| 字段 | 值 |
|---|---|
| `status` | `verified_primary_candidate` |
| `current_app_texture` | 程序生成的示意纹理（Canvas），非科学底图 |
| `source_title` | Mars MGS MOLA DEM 463 m（NASA GSFC / MGS-MOLA，USGS 发布） |
| `source_url` | `https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m` |
| `download_or_service_url` | `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`；彩色版本用 WMS 图层 `MOLA_color` |
| `resolution` | 463.08 m/像素（128 ppd）；导出建议 4096×2048 |
| `file_format` | GeoTIFF（16 bit）→ PNG/JPEG 球体贴图 |
| `map_projection` | Simple Cylindrical |
| `longitude_direction` | Positive East，−180…180 |
| `longitude_range` | −180…180（与 APP 一致，无需平移） |
| `latitude_convention` | Planetocentric（产品页明示） |
| `attribution` | 见第 5 节 |
| `resource_use_terms` | CC0 公有领域，无使用限制 |
| `APP_mapping_check` | 待梁先楷联调：修正 90° 偏移后，用 `ls_0493`、`ls_0592` 验证落点 |

## 7. 下一步

1. 梁先楷：按第 3 节改 `MarsGlobe.vue`，先接 4096×2048 的 MOLA 彩色图；用 `ls_0493`、`ls_0592` 验证经度方向与 90° 偏移。
2. 滕佳杰：纬度核准后判断底图是否需要整体校正；**先不要改底图**。
3. 崔亚鹏：确认 Murray Lab CTX 图幅的引用要求；若要换光学影像，再核 MDIM 2.1 的许可声明。
