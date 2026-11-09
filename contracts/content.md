# 图鉴、案例、资料与资源接口

版本：1，2026-11-09。目录API继续使用 `contracts/catalog.md` 的字段；本组接口读取同一个 `MARS_DB_PATH`，共用其 `dataMode` 和 `version`。图鉴文本版本另以 `contentVersion` 返回，不冒充目录坐标版本。

|GET路径|用途|
|---|---|
|`/api/atlas`|已发布的四类图鉴、类型对比及判读限制|
|`/api/atlas/{SL或SP或FL或CL}`|单类图鉴，包含文本及来源定位|
|`/api/cases?type=SL&page=1&pageSize=20`|已发布案例，支持类型筛选与分页；total在分页前计算|
|`/api/cases/{id}`|案例正文、目录记录、文献引用、影像地址及几何读取地址|
|`/api/documents`、`/api/documents/{id}`|文献标题、HTTPS来源、引用说明；不返回文献本地路径|
|`/api/assets/{id}`|资源类型与预览地址，不返回文件系统路径|
|`/api/assets/{id}/content`|登记且属于当前目录的PNG/JPEG预览；不下载整幅影像、SHP或PDF|
|`/api/landslides/{id}/geometry?kind=polygon`|单个已登记的边界要素；`kind=point`读取点位|

列表的分页上限100，非法类型或参数返回422；不存在或草稿案例返回404。未配置SQLite时，图鉴、案例、文献列表返回空集合，单条返回404，不把演示目录冒充真实案例。

## 内容导入

入口 `scripts/import_atlas_sqlite.py`，接受内容负责人PR #12中的 `atlas.json` 结构：version、sourceDocument、types、cases、comparisons、limits。PR #12合并或组员私下交付文本后，指定实际文件路径：

```powershell
python scripts/import_atlas_sqlite.py --database output/catalog.sqlite3 --input PATH_TO_ATLAS_JSON
# 正文及引用经过审核后，才显式发布：
python scripts/import_atlas_sqlite.py --database output/catalog.sqlite3 --input PATH_TO_ATLAS_JSON --publish
```

导入要求所有案例编号已存在、类别相符、引用可关联，整批事务失败时回滚。默认新条目为draft；重复导入保留已有图片关联和发布状态，不会把ready案例降回draft。内容导入不会写入目录经纬度，也不会把staging目录变成ready。

结构化正文写入新增的可选 `atlas_content` 表，类型定义及案例摘要仍存原表，文献与定位关联继续用documents/case_documents。导入器可给既有schema v1数据库补建此表；旧库没有该表时目录接口照常工作。metadata记录内容版本、文件SHA-256、导入时间与发布操作。启动服务前导入，修改内容后重启，避免前后端缓存使用不同版本。

## 资源及坐标约定

配置 `MARS_ASSET_ROOT` 为私有交接包根目录，登记路径必须实际落在该目录内。预览最多16MiB、16777216像素，只允许实际可解码的PNG/JPEG，有登记SHA-256时须匹配。资源根目录未配置返回503，文件缺失或越出根目录返回404，哈希变化返回409。

几何接口按geometry_links只读取一个SHP要素，核对记录索引及原始编号，不返回整个矢量文件。回复包含 `feature`（采用GeoJSON几何结构）和明确的火星 `coordinateSystem`；不是地球WGS84 GeoJSON，不可套用地球坐标偏移。最多200000个顶点。

每个几何资源的 `assets.metadata_json.latitudeConvention` 默认 `unverified`，原坐标原样返回，`canOverlay=false`。必须在核准该资源的坐标依据后登记为 `planetocentric`，才会返回 `canOverlay=true`。目录已核准不自动证明旧SHP也核准，原PRJ中的WGS84文字也不会触发转换。手机页面应先检查canOverlay再叠加地图；若未核准，展示明确状态。

## 交接边界

梁先楷使用上述HTTP接口接入详情影像、几何开关和地图跳转；滕佳杰维护正文及来源，审核后导入；周璇可使用文献和案例接口中的引用定位，既有Agent资料索引仍按其模块配置更新。本次未替代真实对话或Android真机联调。

完整目录、图片索引、数据库、SHP、影像与组员交接包均不提交GitHub。公开工程只提交接口实现、合成数据测试与运行说明。
