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
| `POST /api/landslide/classify` | 配置真实权重后创建四分类任务，返回 `202`；未配置时 `503` |
| `POST /api/recognition/sessions` | 创建识别记录的设备凭据 |
| `GET /api/recognition/tasks`、`GET /api/recognition/tasks/{id}` | 查询当前设备的记录与结果 |
| `GET /api/recognition/tasks/{id}/image`、`DELETE /api/recognition/tasks/{id}` | 授权预览与删除已结束任务 |
| `GET /api/atlas`、`GET /api/atlas/{type}` | 已发布图鉴与类型对比 |
| `GET /api/cases`、`GET /api/cases/{id}` | 案例、目录记录、引用及资源链接 |
| `GET /api/documents`、`GET /api/documents/{id}` | 文献来源及引用说明 |
| `GET /api/assets/{id}`、`GET /api/assets/{id}/content` | 资源元数据及受限预览图 |
| `GET /api/landslides/{id}/geometry` | 已登记的单个点／面几何及坐标状态 |

图鉴导入、资源根目录配置与坐标状态见 [内容接口约定](../contracts/content.md)。JSON演示模式返回空内容列表；真实素材继续私下交接。

默认将已允许公开的8条真实记录及图片映射为`reference`目录，纬度字段为`unverified, degrees`，原值返回、不做坐标转换；8张图片可从`/samples/<id>.png`读取。参考模式不等于正式坐标已核准的目录。设置`MARS_CATALOG_PATH`可读取其他明确声明模式的本地JSON；目录在启动时校验，修改后需重启。演示数据只在显式指定`examples/catalog.demo.json`时使用。不存在的记录返回404，非法查询参数返回422。

也支持SQLite，字段和现有JSON接口保持一致：

```powershell
python scripts/import_catalog_sqlite.py --input examples/catalog.demo.json --output output/catalog_demo.sqlite3
$env:MARS_DB_PATH = (Resolve-Path output/catalog_demo.sqlite3).Path
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`MARS_DB_PATH`在未显式传入目录路径时优先于JSON环境配置。目录在启动时只读加载，数据库变更后重启服务。数据库结构见`api/db/schema.sql`；`recognition_tasks`仅预留存储，不代表已实现检测任务接口。

原始真实目录用`--raw`导入，状态固定为`staging`，不能直接作为APP目录发布。必须先核准火星坐标约定，再从经过校验的Catalog契约生成独立正式版本；不要仅手工改状态。当前本机原始PRJ写地球WGS84，与火星资料语义不一致，交接记录已标注。

`api/image_policy.py`提供PNG／JPEG输入检查：8MiB、4096×4096总像素上限、最长边8192像素。分类服务复用此入口，任务协议和预处理见 [识别接口约定](../contracts/recognition.md)。

## 配置真实四分类

使用单独推理环境，安装基本依赖后再安装 `api/requirements-classifier.txt`。该文件记录已验证版本；CUDA 设备需安装与本机驱动匹配的 PyTorch wheel，CPU 默认可运行。不需要训练数据来启动服务。

将私下交接的 DINOv2-base 目录、LoRA 适配参数和标准化参数放在仓库外或忽略的 `output/` 下。以下路径是占位示例：

```powershell
python scripts/configure_classifier.py --base /private/dinov2-base --adapter /private/adapter.pt --normalizer /private/normalizer.npz --version frozen-reference-v1 --output output/runtime/classifier-manifest.json
$env:MARS_CLASSIFIER_MANIFEST = (Resolve-Path output/runtime/classifier-manifest.json).Path
$env:MARS_RECOGNITION_DB = "$PWD/output/runtime/recognition.sqlite3"
$env:MARS_CLASSIFIER_DEVICE = 'cpu'
$env:MARS_CLASSIFIER_THREADS = '2'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Manifest 固定模型版本、预处理和四个资源哈希，不上传 Git；资源必须由维护者从可信的私下交接包配置，HTTP 请求不能指定模型路径。设置 `MARS_CLASSIFIER_DEVICE=cuda` 可使用可用的 GPU，默认仅用 CPU 两线程。服务启动校验文件并加载模型，不联网下载。

`MARS_RECOGNITION_DB` 必须与 `MARS_DB_PATH` 的地质目录分开。仅设置任务库而不设置模型时，可以读取历史记录；也可用于模型维护期间的查询。任务库内含上传预览和访问凭据哈希，需按私有数据备份，不放入静态文件目录或版本库。

本版设备凭据用于记录隔离，创建凭据的接口未接账号或公网配额系统。默认只绑定本机；若部署到公网，需由网关提供 HTTPS、入口鉴权和请求配额。不要把演示服务直接当成公开多用户平台。

`MARS_CORS_ORIGINS` 可设置允许访问的前端来源，多个来源用逗号分隔。默认允许本机 Vite 与 Capacitor 页面。手机浏览器联调时可将服务绑定到 `0.0.0.0`，使用电脑局域网地址，并添加实际页面来源；Android封装APP默认使用HTTPS服务，不默认放行明文请求。

测试：

```sh
python -m pip install -r api/requirements-dev.txt
python -m pytest api/tests -q
```
