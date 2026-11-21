# Mars-KG-Vis

火星滑坡探索移动端，支持火星球浏览、目录查询、类型图鉴与点位收藏。

[在线课程演示](https://alanspace.cn/mars/)展示完整编目、四类图鉴和统计图，并接入单图识别与资料问答。仓库按公开样例范围提供可独立运行的本地版本。

仓库另保留滑坡检测与四分类的[模型研究代码及结果说明](docs/research/README.md)，包括人工修框、数据导出、训练对照与联合评估。配置本机模型服务后，客户端可提交单滑坡图片、查询四类分数和回看识别记录；权重单独交接。

组内计划：[分工与八周计划](分工.md)，包含当前进度、每位成员的本周交付及后续安排。

## 当前功能

- 火星球旋转、缩放、点位选择与定位；配置底图后使用经核对的全球影像，加载失败时退回示意纹理或坐标示意。
- 详情影像可缩放、拖动与复位（滚轮、双指、双击、按钮）。
- 已有边界几何按需从现有几何接口加载，可在地图上显示或隐藏；坐标系未确认或记录无几何时明确说明原因。
- 按坐标定位：输入经纬度旋转火星，500 km 内有记录时同时选中该记录。
- 点位定位与列表联动，自动翻到对应页并高亮指定记录。相距不足 1.5° 的密集点位可依次切换。
- 按编号、名称和说明查询，支持类别筛选和翻页。统计页可显示后端用 Matplotlib 绘制的环形图与纬度柱形图，本地目录也支持统计和筛选。
- 真实目录与示例数据在页面分开标识；`#/?focus=<记录id或原始编号>` 可直接定位点位。
- SL、SP、FL、CL 四类目录与案例入口。
- 本机收藏与回访，收藏按最近顺序排列；失效收藏可一键清理。
- 单图识别先筛查滑坡或背景，再对通过筛查的裁片判断类型。历史记录支持回看和删除。
- 上传前校验 PNG/JPEG 裁图（不超过 8 MB、总像素不超过 4096×4096、最长边不超过 8192），提交后显示等待秒数；失败按错误码说明原因。
- 8张公开真实裁片的浏览、缩放、下载和分类试用入口。

本地默认展示 8 条已允许公开的真实滑坡及其影像，前端与 API 使用同一子集。球面按原始坐标绘制点位，坐标及几何校验状态见[编目接口](contracts/catalog.md)。当前线上识别页面接收裁好的单张影像。仓库的 CTX 裁图检测接口配置说明见 [API 文档](api/README.md)。

## 技术栈

移动端使用 Vue 3、TypeScript、Vite、Three.js 和 Capacitor；API 使用 FastAPI。

## 数据采集和转换

在仓库根目录运行以下命令，依次定位 GeoMars 首页、地图脚本和编目资源，再将 493 号记录导出为 CSV 和 JSON。

```bash
python data/scripts/download.py --source-id 493 --output-dir output/geomars-demo
```

`data/scripts/download.py` 使用 Python 标准库读取资源。`scripts/export_catalog.py` 将编目 CSV 转为接口所需字段，核验 ID、坐标范围及坐标状态。导出的文件保存在本地输出目录。

## 本地运行

需要 Node.js 22.12+、Python 3.11+。

```bash
cd mobile
npm ci
npm run dev
```

浏览器打开 `http://localhost:5173`。默认目录和收藏无需后端或在线地图服务。

如需联调API，在仓库根目录运行：

```bash
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

将 `mobile/.env.example` 复制为 `mobile/.env.local`，设置：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

重启前端后从API读取目录。原生APP连接远端服务时使用HTTPS；手机上的127.0.0.1指向手机自身。

网页识别凭据仅保存在`sessionStorage`，刷新后可继续查看当前会话的记录；关闭标签页后凭据会清除，旧记录不会自动恢复。图片与结果保存在配置的服务端，收藏仍在本机。原生APP跨启动历史需接入设备安全存储后再验收，目前不宣称已实现账号或跨设备同步。若使用其他开发端口，应将该精确来源加入API的`MARS_CORS_ORIGINS`。

## SQLite

API支持JSON和SQLite。建立演示数据库：

```bash
python scripts/import_catalog_sqlite.py --input examples/catalog.demo.json --output output/catalog_demo.sqlite3
```

设置`MARS_DB_PATH`为数据库绝对路径后启动API。原始真实目录导入后保持staging，坐标核准前不会作为正式目录发布。结构及约定见[数据库说明](contracts/database.md)和[识别接口约定](contracts/recognition.md)。真实四分类的启动配置见[API说明](api/README.md)，本周交付见[第4周说明](docs/week4/README.md)。

## 智能助手（Agent）

问候、常见编号查找、地图定位和统计直接读取编目。`POST /api/agent/chat` 返回完整答复，`POST /api/agent/chat/stream` 支持流式答复。开放式问题需在服务端配置兼容的语言模型，例子如下。

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=<你的服务端密钥>
```

`MARS_DOCS_PATH` 可指向自备的图鉴或案例资料 JSON。未配置语言模型时，开放式问题返回 `llm_unavailable`，目录快捷查询仍可用。

用户也可以在助手的模型设置中选择服务，填写模型 ID 和自己的 API Key。网页直接请求用户选择的 HTTPS 服务，密钥留在当前页面内存中。项目后端为编号查询和地图定位提供结构化结果，为开放式问题提供公开图鉴摘录。自定义服务需要允许当前网页来源跨域调用。

## Android

Android工程位于 `mobile/android/`。需要 JDK 21、Android SDK 36及对应构建工具。

```bash
cd mobile
npm run android:sync
npm run android:open
```

在Android Studio中运行到手机，或使用Gradle构建调试安装包：

```bash
cd android
./gradlew assembleDebug
```

Windows使用 `gradlew.bat assembleDebug`。输出位于 `app/build/outputs/apk/debug/`。

也可在GitHub Actions中手动运行 `Android debug build`，完成后下载 `mars-atlas-debug` 安装包产物。

## 检查

```bash
cd mobile
npm test
npm run build
```

API测试在仓库根目录运行：

```bash
python -m pip install -r api/requirements-dev.txt
python -m pytest api/tests -q
python -m pytest data/scripts/test_download.py -q
```

## 目录

```text
mobile/                  移动端页面、组件与Android工程
api/                     目录接口与模型服务入口
examples/                公开演示目录
contracts/               数据格式与接口说明
scripts/                 本地目录转换工具
.github/workflows/       构建与接口检查
data/scripts/            模型研究、数据构建与审核工具
tests/                   数据与审核流程测试
tools/maintenance/       研究结果回算与本地维护入口
```

仓库默认展示[8条真实字段与裁片](examples/README.md)，另保留虚构测试数据。完整目录、SHP、其余研究影像、完整人工标注、模型权重和访问凭据单独交接，不随源码公开。真实记录以`reference`模式展示，未核准坐标不能声明为正式`catalog`模式。

目录按[数据格式](contracts/catalog.md)接入；实际目录HTTP样例见[接口样例](contracts/examples/catalog_api.demo.json)。提交前可运行`python scripts/check_public_assets.py`检查公开数据范围。
