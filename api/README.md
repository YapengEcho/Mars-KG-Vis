# API

在仓库根目录执行：

```sh
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

接口文档：`http://127.0.0.1:8000/docs`。

| 接口 | 功能 |
| --- | --- |
| `GET /api/health` | 服务状态与目录版本 |
| `GET /api/catalog` | 完整目录、坐标说明与版本，供客户端一次加载 |
| `GET /api/landslides` | 目录查询，支持 `q`、`type`、`page`、`pageSize` |
| `GET /api/landslides/{id}` | 记录详情 |
| `GET /api/stats` | 类型统计，支持与列表相同的 `q`、`type` |
| `GET /api/capabilities` | 分类、检测服务可用状态 |
| `POST /api/landslide/classify` | 分类入口，当前返回 `503 MODEL_NOT_CONFIGURED` |

默认读取 `examples/catalog.demo.json`，所有记录均为合成演示数据。设置环境变量 `MARS_CATALOG_PATH` 可读取同结构的本地 JSON；目录在服务启动时校验，修改后需重启。经纬度采用目录声明的火星坐标，原值返回。不存在的记录返回 404，非法查询参数返回 422。

也支持SQLite，字段和现有JSON接口保持一致：

```powershell
python scripts/import_catalog_sqlite.py --input examples/catalog.demo.json --output output/catalog_demo.sqlite3
$env:MARS_DB_PATH = (Resolve-Path output/catalog_demo.sqlite3).Path
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`MARS_DB_PATH`在未显式传入目录路径时优先于JSON环境配置。目录在启动时只读加载，数据库变更后重启服务。数据库结构见`api/db/schema.sql`；`recognition_tasks`仅预留存储，不代表已实现检测任务接口。

原始真实目录用`--raw`导入，状态固定为`staging`，不能直接作为APP目录发布。必须先核准火星坐标约定，再从经过校验的Catalog契约生成独立正式版本；不要仅手工改状态。当前本机原始PRJ写地球WGS84，与火星资料语义不一致，交接记录已标注。

`api/image_policy.py`提供PNG／JPEG输入检查：8MiB、4096×4096总像素上限、最长边8192像素，返回明确错误码。该模块已单独测试，目前公开识别入口仍返回503，没有假装已接入模型；真实推理的预处理和任务超时见`contracts/recognition.md`。

`MARS_CORS_ORIGINS` 可设置允许访问的前端来源，多个来源用逗号分隔。默认允许本机 Vite 与 Capacitor 页面。手机浏览器联调时可将服务绑定到 `0.0.0.0`，使用电脑局域网地址，并添加实际页面来源；Android封装APP默认使用HTTPS服务，不默认放行明文请求。

测试：

```sh
python -m pip install -r api/requirements-dev.txt
python -m pytest api/tests -q
```
