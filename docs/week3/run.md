# 第 3 周运行说明

## 1. 环境

- Node ≥ 22.12（本机用 `D:\python_project\github\tools\node` 的 v24.22.0）。
- Windows 下 `%TEMP%` 不可写时，先 `$env:TEMP = 'D:\python_project\.tmp'; $env:TMP = $env:TEMP`，
  否则 vitest/vite 会报 `EPERM mkdir ...\ssr`。
- Python 3.12（venv `D:\python_project\.venv-api`）仅用于导出脚本，无第三方依赖。

## 2. 跑起来

```powershell
$env:PATH = 'D:\python_project\github\tools\node;' + $env:PATH
cd D:\python_project\github\Mars-KG-Vis\mobile
npm ci          # 首次；本机需 --cache D:\python_project\.npm-cache
npm run dev     # http://localhost:5173/#/atlas
```

页面路径：`#/atlas` → `#/atlas/type/SL` → `#/atlas/case/case-ls-0493` → `#/atlas/compare`。

## 3. 测试与构建

```powershell
cd mobile
npm test          # vitest：33 个用例（catalog 9 + atlas 12 + 页面渲染 5 + 其它 7）
npm run build     # vue-tsc --noEmit && vite build
```

CI（`.github/workflows/checks.yml`）跑 `npm ci` / `npm test` / `npm run build` / `npx cap sync android`
以及 `scripts/check_public_assets.py`；本地 `format:check` 在 Windows 检出上因 CRLF 全量失败，CI 不跑这一项。

## 4. 导出结构化数据

```powershell
cd D:\python_project\github\Mars-KG-Vis
python scripts/export_atlas_content.py --check
python scripts/export_atlas_content.py --out D:\python_project\.tmp\week3\atlas `
  --catalog D:\python_project\teng_jiajie_week1_20261024\01_catalog\mars_landslides.json
```

输出见 `agent.md` 第 2 节；`--out` 省略时等于 `--check`。产物只放交付目录，**不要提交仓库**：
带经纬度的 JSON/CSV 会被 `check_public_assets.py` 拦下（`.csv` 直接禁止，`.json` 要求只含 8 个公开 source id）。

## 5. 本周交付目录

`D:\python_project\data\第3周_2026-11-02_11-08\`

```
00_本周交付索引.md
01_图鉴信息架构.md          ← docs/week3/atlas.md
02_案例清单.md              ← docs/week3/cases.md
03_结构化数据说明.md        ← docs/week3/agent.md
04_atlas-content.json
04_atlas-cases.csv
04_atlas-citations.csv
04_atlas-search.jsonl
04_atlas-export-summary.json
04_atlas-limits.md
05_运行说明.md              ← docs/week3/run.md
README.md
scripts/export_atlas_content.py
screenshots/                ← 可选，页面截图
```

## 6. 已知限制

- 目录为示例/公开子集时，12 例中只有 8 例能关联到记录；其余 4 例只显示图鉴文案（页面会说明）。
- 预览裁片逐张不同尺度（19.3–198.4 m/像素），页面把尺度直接写在标题下，避免读者跨尺度比较纹理。
- 地图联动、详情页几何切换属梁先楷负责；后端 `/api/atlas` 属崔亚鹏负责，当前前端读本地内容文件。
