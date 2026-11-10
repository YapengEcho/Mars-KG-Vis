# Mars-KG-Vis

火星滑坡探索移动端，支持火星球浏览、目录查询、类型图鉴与点位收藏。

仓库另保留滑坡检测与四分类的[模型研究代码及结果说明](docs/research/README.md)，包括人工修框、数据导出、训练对照与联合评估。配置本机模型服务后，客户端可提交单滑坡图片、查询四类分数和回看识别记录；权重单独交接。

## 当前功能

- 火星球旋转、缩放、点位选择与定位；配置底图后使用经核对的全球影像，加载失败时退回示意纹理或坐标示意。
- 详情影像可缩放、拖动与复位（滚轮、双指、双击、按钮）。
- 已有边界几何按需从现有几何接口加载，可在地图上显示或隐藏；坐标系未确认或记录无几何时明确说明原因。
- 按坐标定位：输入经纬度旋转火星，500 km 内有记录时同时选中该记录。
- 点位定位与列表联动：自动滚动并高亮对应卡片；相距不足 1.5° 的密集点位可依次切换。
- 按编号、名称和说明查询，类型筛选，完整结果列表分批加载，并提供加载状态与失败重试。
- 真实目录与示例数据在页面分开标识；`#/?focus=<记录id或原始编号>` 可直接定位点位。
- SL、SP、FL、CL 四类目录与案例入口。
- 本机收藏与回访，收藏按最近顺序排列；失效收藏可一键清理。
- 单图分类上传、任务等待、四类分数与待复核提示；历史记录支持回看和删除。
- 8张公开真实裁片的浏览、缩放、下载和分类试用入口。

默认附带20条虚构点位与本地火星底图；8张真实裁片在示例栏单独展示，不把未核准的真实坐标叠加到球面。分类和资料问答需配置相应服务资源；大图检测尚未接入，不返回模拟预测。

## 技术栈

移动端使用 Vue 3、TypeScript、Vite、Three.js 和 Capacitor；API 使用 FastAPI。

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

对话接口为 `POST /api/agent/chat`，默认未启用，需在服务端配置语言模型（凭据不进仓库）：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

可选：`MARS_DOCS_PATH` 指向图鉴 / 案例资料 JSON（格式见 `api/data/docs/guidebook.demo.json`）。未配置 Key 时，`/api/capabilities` 中 `agent.available=false`，对话接口返回 `llm_unavailable` 而不是报错。

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

仓库提供20条虚构演示记录及[8条真实字段与裁片样例](examples/README.md)。完整目录、SHP、其余研究影像、完整人工标注、模型权重和访问凭据单独交接，不随源码公开。真实样例的坐标口径待核准，不直接作为APP运行目录。

目录按[数据格式](contracts/catalog.md)接入；实际目录HTTP样例见[接口样例](contracts/examples/catalog_api.demo.json)。提交前可运行`python scripts/check_public_assets.py`检查公开数据范围。
