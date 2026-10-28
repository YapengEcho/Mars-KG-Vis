# 目录与接口

## 目录

`examples/catalog.demo.json` 为完整示例。外层字段：

| 字段 | 含义 |
|---|---|
| `version` | 数据版本 |
| `mode` | `demo` 或 `catalog` |
| `coordinateSystem` | 火星、东经为正、经度−180～180、星心纬度（度） |
| `records` | 滑坡记录数组 |

每条记录包含：

| 字段 | 类型及约定 |
|---|---|
| `id` | 唯一字符串，保留稳定编号 |
| `sourceId` | 原始编号；未知为 `null` |
| `name` | 展示名称 |
| `type` | `SL`、`SP`、`FL`、`CL` |
| `longitude` / `latitude` | 火星经纬度，不能套用地球定位转换 |
| `areaKm2` | 平方公里；未知为 `null` |
| `summary` | 有来源的简要说明；未知可留空 |
| `imageUrl` | HTTPS或同源绝对路径；未知为 `null` |
| `source` | `{ "title": "来源名称", "url": null }`；链接使用HTTPS |
| `isDemo` | 示例为 `true`，真实目录为 `false` |

`coordinateSystem` 固定为：

```json
{
  "body": "Mars",
  "longitude": "east-positive, -180..180",
  "latitude": "planetocentric, degrees"
}
```

同一目录不混合示例和真实记录；缺失面积、图片、原始编号不以虚构数值补齐。原始标签和展示标签的映射应在导出时记录。

## 接入

静态目录放到 `mobile/public/data/catalog.local.json`，在 `mobile/.env.local` 设置 `VITE_CATALOG_URL=/data/catalog.local.json`。此路径优先于API。该本地文件不提交Git。

API读取目录使用环境变量 `MARS_CATALOG_PATH`；前端设置 `VITE_API_BASE_URL` 后请求 `GET /api/catalog`。未设置目录和API时使用内置演示目录；加载失败时明确提示当前显示示例，并提供重试入口。

APP的搜索字段与 `GET /api/landslides` 的 `q` 一致：记录 `id`、原始编号 `sourceId`、名称与说明；类型筛选独立。同一查询在APP、API与Agent中返回相同记录。

页面通过 `#/explore?focus=<记录id或原始编号>` 定位点位；编号不存在或目录未加载时在页面内提示，不静默失败。目录模式在页面显示为“真实目录”，示例模式显示“示例数据”，二者不混用。

转换已有JSON：

```bash
python scripts/export_catalog.py --input PATH_TO_CATALOG --output mobile/public/data/catalog.local.json --source-title "资料来源" --coordinate-system planetocentric --limit 20
```

只有原始资料的坐标约定已核准后才使用该转换；它不会进行投影、纬度类型转换或自动判定类别。没有可信图片时保持空值。

## 接口

| 方法与路径 | 返回 |
|---|---|
| `GET /api/health` | 服务及目录状态 |
| `GET /api/catalog` | 完整目录 |
| `GET /api/landslides?q=&type=&page=1&pageSize=20` | `items`、`total`、`page`、`pageSize`、`dataMode`、`version` |
| `GET /api/landslides/{id}` | 单条记录；不存在返回404 |
| `GET /api/stats?q=&type=` | 同条件总数与 `byType` |
| `GET /api/capabilities` | `classifier`、`detection` 的可用状态 |
| `POST /api/landslide/classify` | 当前返回503 `MODEL_NOT_CONFIGURED` |

后续单图分类使用multipart字段 `file`；成功结果约定包含 `prediction`、`probabilities`、`confidence`、`uncertain` 和 `modelVersion`。检测服务另接任务接口，不把单图分类当作大图检测。
