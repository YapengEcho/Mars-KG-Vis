# 第 6 周移动端联调记录

日期：2026-11-26｜分支：`feat-week6`（基于 `origin/main` `f100945`）
范围：网站、API、Android 版本统一；检测候选框与单图历史回看；无结果、超限、断网、会话失效界面。

## 1. 版本基线

| 项 | 版本／结果 |
| --- | --- |
| 代码基线 | `origin/main` `f100945`（含 PR #25、#26、#27） |
| 移动端 | `mars-atlas-mobile` 0.1.0，Vue 3 + Vite 7 + Capacitor 8 |
| Android | `versionCode 1` / `versionName 0.1.0`，minSdk 24，target/compileSdk 36 |
| 调试包 | `mobile/android/app/build/outputs/apk/debug/app-debug.apk`，5,231,730 字节，SHA-256 `B55D0A02ABD137FBB7E9483AF05E942E54F635FD96F8048BCFC6382A19F4437E` |
| 构建工具 | Node 24.21.0、npm 11.19.0、JDK 21、Gradle 8.14.3；Windows 中文路径下需 `android.overridePathCheck=true`（已写入 `mobile/android/gradle.properties`） |
| 移动端测试 | 85 项通过（新增断网提示 1 项） |
| 生产构建 | `npm run build` 通过 |
| API 测试 | 186 项通过（`python -m pytest api/tests -q`） |
| 部署服务 | `geomars-reference-20261120`；分类可用（`dinov2-lora-full-seed20261015-epoch8`），**检测未配置** |

## 2. 网页端实测

### 2.1 检测链路（接口契约一致的本机桩服务）

本机未配置检测权重，部署服务的 `/api/capabilities` 也返回 `detection.available=false`。为验证页面与接口契约，使用 `FastAPI create_app()` 注入固定结果的桩分类器／检测器，接口、鉴权、任务库、预览图与轮询均走真实代码路径；**桩分数不代表模型效果**。

| 场景 | 操作 | 结果 | 观察耗时 |
| --- | --- | --- | --- |
| 正常检出 | 1800×1500 PNG 上传 | 2 个候选、SVG 2 个框，预览 1800×1500 与输入一致 | 提交 `202` 约 64 ms；点击到候选显示约 1.6 s |
| 候选点选 | 点「候选 1」 | 像素框 `[216, 225, 756, 720]`、检测 93%、四类分数与「查看全图」出现，缩放 `scale(3.03)` | 即时 |
| 历史回看 | 「我的」→「裁图检测」→ 打开记录 | 回到同一任务 `483ab44e…`，2 个框、1800×1500 预览 | 列表约 2 s 内返回 |
| 未检出 | 纯黑 800×600 PNG | 「未检出候选 · 待复核」+「本次没有检出的候选…」0 个框 | 约 1.1 s |
| 超限 | 2400×1200 PNG | 未提交，提示「最长边不超过 2048…」，提交按钮禁用 | 即时 |
| 会话失效 | 重建任务库后保留旧凭据刷新 | 「本次会话凭据已失效…」+「重新建立会话」 | 首次查询即返回 401 |
| 断网 | 停掉本机 API 后刷新 | 「无法连接识别服务，请检查网络后重试。」 | 即时失败 |
| 重连 | 重启 API 后点「重新建立会话」 | 提示消失，回到空记录状态 | 约 1.8 s |

### 2.2 单图分类真实任务（部署服务）

部署服务未对本地 Vite 来源返回 CORS 头，联调时用本机反向代理转发 `/api`（仅测试基础设施，不改应用代码）。三张公开样例各跑一次：

| 样例 | 任务 | 结果 | 提交到最终状态 |
| --- | --- | --- | --- |
| SL 493 | `d6eff7e5…` | 疑似滑坡 · 待复核；SL 44.4%、CL 37.9% | 约 3 s 内 |
| SP 483 | `932fe4a9…` | 疑似滑坡 · 待复核；SP 94.8% | 结果页约 1.5 s，任务约 5.5 s 内完成 |
| FL 597 | `19ef729d…` | 疑似滑坡 · 待复核；FL 92.0% | 3.2 s |

「我的」→「单图分类」可看到三条记录及类型标签；点开 `19ef729d…` 回到同一任务与相同分数。

### 2.3 本周修复

- `mobile/src/services/recognition.ts`：捕获 `fetch` 网络失败与超时，分别映射为「无法连接识别服务，请检查网络后重试。」和「请求超时，请检查网络后重试。」，不再向界面抛英文 `Failed to fetch`。
- `mobile/src/services/recognition.test.ts`：新增网络失败映射用例。
- `mobile/android/app/src/debug/AndroidManifest.xml`：仅调试包允许明文 HTTP，便于真机连局域网 API；release 仍保持 HTTPS。

## 3. 未完成项与原因

| 项 | 状态 | 原因／下一步 |
| --- | --- | --- |
| Android 真机回归 | **未完成** | 当前环境 `adb devices` 无设备，也无模拟器；需接入一台 Android 手机后按第 4 节执行 |
| 真实检测任务（候选框） | **未完成** | 本机无私有权重，部署服务未配置检测；需崔亚鹏提供检测服务地址或按 `api/README.md` 配置 `MARS_DETECTOR_*` |
| 真机断网／重开／凭据 | **未完成** | 依赖真机；网页端已覆盖对应状态 |

因此本记录只能证明：网页端检测与单图流程、异常界面和单图真实模型可用；不能宣称真机与真实检测已完成。

## 4. 真机验收步骤（待执行）

1. 启动可被手机访问的 API（同一局域网）：

   ```powershell
   $env:MARS_RECOGNITION_DB = "$PWD/output/runtime/recognition.sqlite3"
   $env:MARS_CORS_ORIGINS = "http://localhost,https://localhost,capacitor://localhost"
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```

2. 用电脑局域网地址构建联调包，再安装：

   ```powershell
   cd mobile
   $env:VITE_API_BASE_URL = "http://<电脑局域网 IP>:8000"
   npm run android:sync
   cd android
   $env:JAVA_HOME = "C:\dev\jdk21"
   .\gradlew.bat assembleDebug
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   ```

3. 依次记录（每条保留截图或录屏）：

   | 检查 | 预期 |
   | --- | --- |
   | 五个入口 | 探索、图鉴、识别、助手、我的均可打开 |
   | 检测上传 → 候选 | 与 2.1 相同：框、分数、点选放大、查看全图 |
   | 检测历史 | 关闭重开 APP 后同设备凭据仍可回看（若未实现需在记录中注明） |
   | 单图历史 | 分类记录与检测记录分标签，不串类 |
   | 未检出 | 显示「本次没有检出的候选」，不是失败页 |
   | 超限 | 2400×1200 在客户端被拦截，不发请求 |
   | 断网 | 飞行模式后提交／刷新，显示可重试中文提示 |
   | 会话失效 | 服务端清库或换库后，显示「重新建立会话」并可恢复 |
   | 重新连接 | 恢复网络后重试成功，不重复提交同一图片 |

4. 记录表字段：手机型号、Android 版本、APP 版本、API 地址与版本、网络类型、每步耗时、截图编号、异常码。

## 5. 复现命令

```powershell
# 网页端
cd mobile
npm test
npm run build

# API
cd ..
python -m pytest api/tests -q

# Android
cd mobile
npm run android:sync
cd android
$env:JAVA_HOME = "C:\dev\jdk21"
.\gradlew.bat assembleDebug
```
