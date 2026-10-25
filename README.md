# Mars-KG-Vis

火星滑坡探索移动端，支持火星球浏览、目录查询、类型图鉴与点位收藏。

仓库另保留滑坡检测与四分类的[模型研究代码及结果说明](docs/research/README.md)，包括人工修框、数据导出、训练对照与联合评估。研究权重尚未接入移动端 API。

## 当前功能

- 火星球旋转、缩放、点位选择与定位；无法显示三维时切换为坐标示意。
- 按编号、名称和类型查询，查看点位详情。
- SL、SP、FL、CL 四类目录与案例入口。
- 本机收藏与回访。
- 单图分类上传页面及服务状态检查。

默认附带20条虚构示例记录，球面纹理为程序生成的示意图。分类模型、大图检测和资料问答尚未接入，不返回模拟预测。

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

## SQLite

API支持JSON和SQLite。建立演示数据库：

```bash
python scripts/import_catalog_sqlite.py --input examples/catalog.demo.json --output output/catalog_demo.sqlite3
```

设置`MARS_DB_PATH`为数据库绝对路径后启动API。原始真实目录导入后保持staging，坐标核准前不会作为正式目录发布。结构及约定见[数据库说明](contracts/database.md)和[识别接口约定](contracts/recognition.md)。识别HTTP入口目前仍返回503。

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

仓库提供20条虚构演示记录及[8条真实字段样例](examples/README.md)。完整目录、SHP、研究影像、人工标注、模型权重和访问凭据单独交接，不随源码公开。真实样例的坐标口径待核准，不直接作为APP运行目录。

目录按[数据格式](contracts/catalog.md)接入；实际目录HTTP样例见[接口样例](contracts/examples/catalog_api.demo.json)。提交前可运行`python scripts/check_public_assets.py`检查公开数据范围。
