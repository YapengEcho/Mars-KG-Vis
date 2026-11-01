# 第 2 周运行说明

## 1. 解释器

`api/` 依赖 pydantic 与 fastapi，系统的 CPython 3.14 与 DSH 自带的 3.12 都没有装；本机用
`D:\python_project\.venv-api` 这个 venv：

```powershell
D:\python_project\.venv-api\Scripts\python.exe -c "import fastapi, pydantic, shapefile; print(fastapi.__version__, pydantic.VERSION)"
# 0.115.14 2.14.2
```

不装这些也能跑准入检查的纯几何部分，但 `--mode ready` 的 `Catalog` 校验会被跳过（脚本会打印提示）。
几何检查需要 `pyshp`（导入名 `shapefile`），venv 里已装 3.1.6。

### 离线装包（pip 在本机会卡死）

这台机器上 `pip install` 会长时间无输出（网络层卡住），`urllib` 直连 PyPI 却是正常的。
因此依赖是先用 `urllib` 下 wheel 到 `D:\python_project\.wheels`，再离线安装：

```powershell
& $py -m pip install --no-input --no-index --find-links D:\python_project\.wheels `
    pydantic==2.14.2 fastapi==0.115.14 uvicorn==0.34.3 pyshp==3.1.6
```

`--find-links` 的匹配要接受 `-none-any.whl`（`colorama` 就是 `py2.py3-none-any`）。
`pillow` 必须用 `<13`（`pyproject` 约束 `Pillow>=10.4,<13`），`openai` 必须 `>=1.50,<2`。

## 2. 命令

```powershell
$py = 'D:\python_project\.venv-api\Scripts\python.exe'
$handoff = 'D:\python_project\teng_jiajie_week1_20261024'
$out = 'D:\python_project\.tmp\week2'
# 1) 准入（默认 unverified：纬度原样保留，结论为 conditional）
& $py scripts/admission_report.py --handoff $handoff --json "$out\admission.json" --md "$out\admission.md"

# 2) 导出 + 建库（staging 即可交付；默认基线就是它）
& $py scripts/export_checked_catalog.py --handoff $handoff --out "$out\staging" --mode staging --skip-sha --sqlite "$out\staging.db"

# 3) 案例映射 + 写库（复制库再写案例）
Copy-Item "$out\staging.db" "$out\cases.db"
& $py scripts/build_case_mapping.py --handoff $handoff --out "$out\cases" --sqlite "$out\cases.db"

# 4) 只有拿到可追溯的纬度依据后，才允许 ready
$evidence = '2026-11-03 老师确认：编目纬度按行星面纬度（planetographic）使用'
& $py scripts/export_checked_catalog.py --handoff $handoff --out "$out\ready" --mode ready `
    --latitude-convention planetographic --coordinate-evidence $evidence --sqlite "$out\ready.db"

# 5) 换算影响（不写库，只打印越界对比）
& $py scripts/compare_latitude_conventions.py --handoff $handoff
```

`build_database` 不覆盖已有文件，重复跑请换文件名或先删。

## 3. 测试

```powershell
$env:TEMP = 'D:\python_project\.tmp'   # 否则 pytest/unittest 的临时目录会撞权限
& $py -m unittest discover -s scripts -p 'test_week2_scripts.py' -v
```

35 项：纯函数 22 项（屏幕坐标→经纬、球面面积、点在多边形、摘要、CSV、行星面↔行星心换算与位移统计）
+ 交接包集成 9 项（准入无 fail、默认无依据时结论为 conditional、有依据且有约定时为 ready、
默认导出不改纬度、人工框 736 点移出、取图框保持自身坐标、条数与类别、12 例无缺项）
+ 外键 4 项（完整映射可写、缺父记录回滚、孤儿案例被拒）。
交接包不存在时集成测试自动跳过，`HANDOFF_DIR` 环境变量可以指向别的位置。

CI 只跑 `api/tests` 与 `scripts/test_export_catalog.py`，本文件的测试目前是本地跑；
如果要进 CI，需要把交接包挂成 artifact 或只跑纯函数部分。

## 4. 已知坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `UnicodeEncodeError: 'gbk' codec can't encode ...` | PowerShell 控制台默认 GBK，脚本里有 `⁻⁶` 之类字符 | 脚本已在 `main()` 里 `sys.stdout.reconfigure(encoding='utf-8')`，新脚本照抄这一行 |
| `FileExistsError: Database already exists` | `build_database` 不覆盖 | 换文件名，或先删旧库 |
| 图幅名报「CTX 网格中无 E-084_N-08」 | 拿 `ctx_grid` 的文件名当图幅名 | 从 shapefile 的 `Name` 字段读 |
| 案例纬度符号反了 | `native_window_world_m` 的 `y` 向南为正 | `lat = −y·180/(πR)` |
| 点位整体偏南/北十几公里 | 行星面纬度当成行星心纬度入库 | 走 `mars_latitude` 换算，别只改标签 |
| 736 个点跑出人工取图框 | 在对纬度约定没有依据时默认做了行星面→行星心换算 | 默认 `unverified` 不换算；对照 `compare_latitude_conventions.py` |
| 案例写库只写了一半 | 没开外键、父记录不存在时静默插入 | 已开 `PRAGMA foreign_keys=ON` + `foreign_key_check` + 单事务回滚 |
| `pip install` 卡住无输出 | pip 网络层异常 | 用 `--no-index --find-links` 离线装 |
| `npm`/`git` 相关命令报 TLS 或执行策略错误 | schannel 异常、PowerShell 脚本受限 | 见第 1 周 `docs/week1/run.md` |
