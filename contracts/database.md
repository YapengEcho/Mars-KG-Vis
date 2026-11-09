# SQLite结构与交接

版本1，结构源文件为`api/db/schema.sql`，导入入口为`scripts/import_catalog_sqlite.py`。使用Python标准库sqlite3，无需独立数据库服务器。

|表|用途|
|---|---|
|metadata|版本、数据模式、坐标约定、release_state和建库时间|
|sources|原始资料来源|
|landslides|稳定id、原始序号、四类展示标签、raw_type、经纬度、面积及原始记录|
|assets|图片、点／面文件、底图、文档的路径与哈希；本地路径不等于公网URL|
|geometry_links|目录编号与点／面资产、原文件零起始记录索引的对应|
|atlas_types|四类图鉴内容；草稿和正式版本分别标记|
|cases|案例及主图片，候选先标draft，不冒充已确认案例|
|documents／case_documents|资料与案例的引用、页码和图号|
|recognition_tasks|识别任务持久化预留；任务接口和访问隔离在后续实现|

业务关联使用`landslides.id`与`source_id`，不拿SHP行号或ORIG_FID当稳定编号。真实边界中59、60为RL，484为DF；现有目录对应SL、SL、FL，保留raw_type与来源。

外键由导入器启用；其他写库工具也需执行`PRAGMA foreign_keys=ON`。原始数据先入staging；只读API加载器拒绝未核准坐标的数据。正式版本由清理并核准的Catalog文件重建，不覆盖历史库。导入器拒绝覆盖既有路径，参数化写入并检查完整性与外键。

本周已有2971条原始记录、1083张图片索引、5942条点／面记录关联。四类图鉴及12个案例只是待补内容的草稿，不表示滕佳杰的内容任务已经完成。后续由滕提供最终内容、来源与坐标依据，崔维护导入、资源服务与数据库版本。

## 第2周导入记录

`scripts/import_catalog_sqlite.py` 支持 `--version`。每次导入新建数据库，拒绝覆盖旧版本；在 `metadata.importProvenance` 中写入输入文件、内容、编号索引、图片索引的SHA-256以及坐标依据。标准输出为可保存的导入凭据，包含版本、分类计数、来源数、资产数、完整性、外键检查和数据库哈希。直接调用 `build_database` 时也记录内容和索引哈希。

准入入口导入真实 `ready` 目录时要求 `--coordinate-evidence`；这段文字必须指向真实依据，不能把参数存在视为已获得专家确认。`--raw` 始终保留原始经纬度，标为staging，API拒绝加载；导入器本身不推测纬度类型、不自动转换。ready导入通过 `source_index` 保留RL、DF等原始类型，展示类别仍沿用原映射。

本轮新建的暂存库只接入目录、来源和1083条图片索引；上一周的5942条几何关联仍保存在原库。未把尚待修正的第2周案例转换产物写入新库。
