# 火星滑坡目录：字段与坐标说明 v1

版本：v1（2026-10-24）｜适用范围：`teng_jiajie_week1_20261024` 交接包（2971 条编目）
｜对应 APP 契约：[`contracts/catalog.md`](catalog.md)
｜校验脚本：[`scripts/handoff_check.py`](../scripts/handoff_check.py)

本文只记录**已核实的事实**和**明确的待确认项**。凡未核实的，一律写“待核准”，不写成结论。

## 1. 数据构成

| 文件 | 内容 | 规模 |
|---|---|---|
| `01_catalog/mars_landslides.json` | 编目主表（每条：编号、类型、面积、周长、经纬度、度分秒文本） | 2971 条 |
| `01_catalog/mars_landslides.csv` | 同上，CSV 版 | 2971 行 |
| `01_catalog/source_id_index.json` | `source_id` ↔ 目录 id ↔ 点位/边界记录下标 | 2971 条 |
| `02_vectors/landslide_points/` | 滑坡点 SHP（PointZ） | 2971 个点 |
| `02_vectors/landslide_boundaries/` | 滑坡边界 SHP（PolygonZ） | 2971 个面 |
| `02_vectors/ctx_grid/` | CTX 4°×4° 图幅分区（Murray Lab CTX 镶嵌） | 3960 个图幅 |
| `03_images/train224/` | 224×224 分类裁片（灰度） | 1083 张 |
| `03_images/image_index.json` | 图片索引（路径、SHA-256、图幅、窗口坐标） | 1083 条 |

### 1.1 类型计数（已核对）

| 展示类型 | 中文 | 英文 | 目录条数 |
|---|---|---|---|
| SL | 滑移型 | slide-type landslide | 1787 |
| SP | 滑塌型 | slump-type landslide | 321 |
| FL | 流动型 | flow-type landslide | 50 |
| CL | 复合型 | complex landslide | 813 |

原始边界属性里的 `RL` 2 条、`DF` 1 条，按既有口径分别归入 `SL`、`FL`：

| source_id | 原始边界类型 | 点位/目录类型 |
|---|---|---|
| 59 | RL | SL |
| 60 | RL | SL |
| 484 | DF | FL |

映射后计数与论文《火星全球滑坡编目及分类初探》3.2 节（PDF 第 7 页）给出的 SL 1787、SP 321、FL 50、CL 813 一致。**这是数量吻合，不等于论文逐条确认了这 3 条别名**；原始类型必须保留在数据字典里，不能因为映射就丢掉。

## 2. 字段表

### 2.1 目录主表 → APP 契约

| 原始字段 | 类型 | 单位/取值 | APP 字段 | 约定 | 缺失处理 |
|---|---|---|---|---|---|
| `id` | string | `ls_0001` … `ls_2971` | `id` | 稳定编号，不按排序重编 | 不允许为空 |
| `source_id` | int | 1…2971，连续 | `sourceId` | 与 SHP `序号` 对应；不是 FID/行号 | 不允许为空 |
| `name` | string | 本批**全部为空** | `name` | 有正式地名才填 | 生成 `滑坡 {source_id}` |
| `type` | string | SL/SP/FL/CL | `type` | 四类展示枚举 | 不允许为空 |
| `type_en` / `type_cn` | string | 英文/中文类型名 | — | 仅用于展示与检索 | — |
| `area` | float | km² | `areaKm2` | 编目面积，不用 224 图像素面积替代 | 允许 null，不补数 |
| `perimeter` | float | km | （不进 APP 必填） | 原编目周长，源单位核准后再写字典 | 允许 null |
| `lon` | float | 度，−180…180，东经为正 | `longitude` | 见第 3 节 | 不允许为空 |
| `lat` | float | 度，−90…90 | `latitude` | 见第 3 节 | 不允许为空 |
| `lon_degree` / `lat_degree` | string | `-106°13′33.88″` | — | 仅作展示；见第 5 节缺号问题 | — |
| `desc` | string | 本批**全部为空** | `summary` | 有出处才写 | 置空字符串 |
| （无） | — | — | `imageUrl` | 真实资源地址待接；`03_images/...` 是本地路径，不是 APP URL | null |
| （无） | — | — | `source` | 必须有 `title`；`url` 用 HTTPS | 不允许为空 |
| （无） | — | — | `isDemo` | 真实目录为 false | 不允许为空 |

外层：`version`（数据版本，建议 `2026-10-19` 起编）、`mode`（`demo` 或 `catalog`）、`coordinateSystem`（固定 Mars / east-positive, −180..180 / planetocentric, degrees）。

### 2.2 SHP 属性

| 文件 | 字段 | 说明 |
|---|---|---|
| 滑坡点 / 边界 | `序号` | 等于 `source_id`，逐条一致（已全量核对） |
| 滑坡点 / 边界 | `类型` | 原始类型，含 RL/DF 别名 |
| 滑坡点 / 边界 | `area`、`Perimeter` | 与主表数值一致；面积单位为 km² |
| 滑坡点 / 边界 | `经度`、`纬度` | 度分秒文本，见第 5 节 |
| 滑坡点 | `ORIG_FID` | 原始要素 ID，仅供追溯 |
| 边界 | 几何 | `PolygonZ`，Z 值全为 0（无高程信息） |

## 3. 坐标约定：经度已核，纬度待核准

### 3.1 已核实

1. **经度范围与方向**：全表 `lon ∈ [−179.8789, 178.4853]`，负值 2444 条；论文图 2（PDF 第 8 页）横轴标注 `180°W … 0° … 180°E`。两者一致，可确认**东经为正、−180…180**。
2. **数值来源一致**：点位 SHP 几何坐标与主表 `lon`/`lat` 在 6 位小数舍入误差内完全一致（2971 条全查，最大差 1.4×10⁻⁶ 度）。
3. **面积/周长用火星球体算的**：SHP 元数据（`.shp.xml` lineage）显示老师用 ArcGIS 的
   `PROJCS["Equirectangular_Mars", GEOGCS["GCS_Mars", DATUM["D_Mars", SPHEROID["Mars_localRadius", 3396190.0, 0.0]]]]`
   计算面积与周长（`AREA_GEODESIC` / `PERIMETER_LENGTH_GEODESIC`）。这是**球体**（扁率 0，半径 3396190 m），与影像裁片窗口坐标（`native_window_world_m`，按 R=3396190 换算经度正好落在 4° 图幅边界）互相印证。
4. **图片窗口同一坐标系**：`native_window_world_m` 与 `manual_box_world_m` 都是火星等距圆柱平面坐标，单位为米。

### 3.2 未核准

`latitude` 究竟是**行星心纬度（planetocentric）**还是**行星面纬度（planetographic）**，现有文件**不能认证**：

- 原始 `.prj` 写的是 `GCS_WGS_1984`（地球椭球 6378137 m、扁率 1/298.257），对火星数据是错误标注，**不能拿它当依据**，也不能据此做地球投影转换。
- 论文正文未声明纬度约定；图 2 只给经纬网。
- 上面第 3 条只能说“面积是按火星球体算的”，球体上没有面纬度可言，但这不能反推**输入坐标**一定是行星心纬度。

### 3.3 影响量化

若实际是行星面纬度而 APP 按行星心纬度渲染，纬度方向最大偏差：

| 统计 | 值 |
|---|---|
| 全表最大偏差 | 0.34° |
| 中位偏差 | 0.10° |
| 45°N/S 附近 | 约 0.25°（≈15 km） |

这个量级在球体缩放到区域时会明显偏点位，**不能忽略，也不能拍脑袋转换**。

### 3.4 核准步骤（待执行，结论回填本文件 v2）

1. 在 NASA Mars Trek 或 ArcGIS Pro 火星底图上取 5 个可识别地标（如 Olympus Mons 峰顶、Gale 撞击坑中心），分别用 planetocentric / planetographic 两套显示对比。
2. 从编目里挑这 5 个地标附近、边界清晰的记录，比较点位与地标边缘的相对位置，记录偏差方向。
3. 向老师索取编目工程的坐标说明或出图设置截图；以上三条任一成立即写“已核准”，否则维持“待核准”。
4. 核准前：真实目录只放本地 `mobile/public/data/catalog.local.json`，**不进默认打包产物**；APP 内标注“坐标约定待核准”。

## 4. 图片与几何的对应

| 关系 | 规则 |
|---|---|
| 目录 ↔ 点位/边界 | 统一用 `source_id`（= SHP `序号`）关联；`*_record_index0` 只是本版文件内的读取下标，不能当业务主键 |
| 目录 ↔ 图片 | `03_images/image_index.json` 的 `source_id`；文件名 `slide_{source_id:05d}.png` |
| 覆盖情况 | 2971 条中 1083 条有裁片（36.5%）：SL 531、SP 94、FL 15、CL 443；其余记录**没有图**，不许用别的记录顶替 |
| 图片规格 | 224×224、灰度（mode `L`）、SHA-256 已逐张核对通过 |
| 裁片窗口 | `native_window_world_m` 宽约 8595 m（≈0.145°），与 4° 图幅原点对齐（如 `E-084` 窗口起点 −84.0000°），**不是以点位为中心**；折算约 38.4 m/像素，与 CTX 原生约 6 m/像素的关系（重采样/降采样方式）待崔亚鹏确认 |
| 人工框 | `manual_box_world_m` 是后来人工训练框，**与编目边界不是同一套几何**，不能互相替代 |
| 边界用途 | 编目多边形，可直接算外接框做检测标签；分割 mask 需栅格化，不能用截图或红色描边代替 |

## 5. 已知数据问题：度分秒文本丢负号

93 条记录的 `经度`/`纬度` 文本在**度为 0** 时丢了负号，例如 `source_id=686`：

| 字段 | 值 |
|---|---|
| 几何 / 主表 `lat` | −0.425023° |
| `纬度` 文本 | `0°25′30.08″`（应为 `−0°25′30.08″`） |

全量核对结果：93 条全部属于“绝对值正确、仅符号丢失”，没有真正的数值错误。

处理规则：**APP 与导出脚本只用数值字段** `lon`/`lat`；度分秒文本仅作展示，且显示时应由数值重新格式化，不要直接引用原文本。

## 6. 版本、校验与复现

- 交接清单：`handoff_manifest.json`（计数、类型映射、来源 SHA-256）；文件校验：`checksums.json`。
- 复现校验：`python scripts/handoff_check.py --handoff <包路径>`，输出 27 项 PASS、5 项 WARN，见 `docs/week1/handoff-check.md`。
- 生成 20 条样例与 12 例清单：`python scripts/build_week1_artifacts.py --handoff <包路径> --out <输出目录>`。

## 7. 待确认清单

| 事项 | 现状 | 需要谁 | 影响 |
|---|---|---|---|
| 纬度约定（planetocentric / planetographic） | 待核准 | 老师 + 崔亚鹏 | 点位偏差最大 0.34°，阻塞真实目录上架 |
| 224 裁片的重采样方式与窗口规则 | 只有窗口坐标，无处理脚本 | 崔亚鹏 | 影响“高清图替换预览图”的一致性 |
| `perimeter` 的单位 | 主表无单位说明 | 崔亚鹏 | 只影响附加字段 |
| 边界是人工解译还是由点/面积生成 | 元数据未说明 | 老师 | 决定能否用于分割验收 |
| 案例典型性 | 12 条是候选 | 滕佳杰 + 老师 | 图鉴内容定稿 |
