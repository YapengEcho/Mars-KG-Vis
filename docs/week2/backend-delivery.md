# 第2周后端与队长验收记录

日期：2026-10-31。对应计划周期：10月26日—11月1日。负责人：崔亚鹏。

本次代码以main `2f721d3416bef9ebbef6378558a64d727f591c96` 为基础，保留已合并的APP和Agent功能。研究目录中的历史代码、模型和训练配置没有修改。

## 已完成

|工作|交付与证据|
|---|---|
|真实目录与来源入库|新版本SQLite保存2971条原始记录、来源、1083条图片索引；完整性正常、外键错误0；保留原始坐标和原始类别|
|列表、详情和同条件统计|沿用统一响应字段，补齐bbox范围筛选、跨180°经线、边界包含和参数错误处理|
|APP接口与Agent查询一致|提取共享查询实现，Agent支持同样范围及翻页；返回一致的数据版本|
|导入与版本追踪|新增版本参数、输入文件及内容哈希、索引哈希、导入凭据；禁止覆盖旧库|
|验收|62项测试、7个子检查通过；演示SQLite的10组查询通过数据库—HTTP—Agent工具一致性检查，含详情和404|
|组员交付把关|已核查待合并PR #9，复跑68项测试及额外反例；审查建议另附|

## 尚未宣告完成的验收项

2971条真实记录的正式坐标发布仍待核准。论文全文检索和数据方法页未找到明确的原始纬度类型；PR #9声称的“老师确认行星面纬度”目前无可核验依据。因此保留staging，不自动转换、不更改原始数据，不把演示接口通过写成真实地图发布完成。

本次一致性验收不调用外部语言模型，也不是Android真机验收。梁先楷仍需确认手机实际服务地址和目录来源一致；周璇需以真实对话验证工具选择及回答与返回值一致。队长负责统一版本和汇总验收证据，不代替成员声称真机或模型效果已通过。

## 使用

在本目录所属工程根目录安装 `api/requirements-dev.txt` 后执行：

```powershell
python scripts/import_catalog_sqlite.py --input examples/catalog.demo.json --output output/catalog-demo-week2.sqlite3 --version demo-week2-v1
python scripts/verify_catalog_integration.py --database output/catalog-demo-week2.sqlite3 --output output/catalog-check.json
$env:MARS_DB_PATH = (Resolve-Path output/catalog-demo-week2.sqlite3).Path
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

查询示例：`/api/landslides?type=SL&bbox=-100,-30,-40,30&pageSize=20`；统计使用完全相同的q、type、bbox。

真实原始目录使用 `--raw --source-index ... --image-index ... --asset-root ... --version ...` 导入。待原始纬度约定经资料或可靠配准核准后，再从未改写的源文件导出新ready目录，记录依据并重建数据库。不要修改staging标记来跳过核准。

完整目录、影像、数据库和实际交接包仍私下交接；本次代码包和补丁不含这些文件。完整数据不纳入本PR；远端合并仍由仓库评审流程决定。
