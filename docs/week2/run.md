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

## 2. 命令

```powershell
$py = 'D:\python_project\.venv-api\Scripts\python.exe'
$handoff = 'D:\python_project\teng_jiajie_week1_20261024'
$out = 'D:\python_project\.tmp\week2'

# 1) 准入
& $py scripts/admission_report.py --handoff $handoff --out $out --coordinate-evidence $null

# 2) 导出 + 建库
& $py scripts/export_checked_catalog.py --handoff $handoff --out "$out\staging" --mode staging --sqlite "$out\staging.db"

# 3) 案例映射 + 写库
& $py scripts/build_case_mapping.py --handoff $handoff --out "$out\cases" --sqlite "$out\cases.db"
```

把 `staging.db` 复制成 `cases.db` 再跑第三步，就能在一个库里同时看到目录和案例。
`build_database` 不覆盖已有文件，重复跑请换文件名或先删。

## 3. 测试

```powershell
& $py -m unittest discover -s scripts -p 'test_week2_scripts.py' -v
```

18 项：纯函数 13 项（坐标换算、球面面积、点在多边形、摘要、CSV 生成）+ 交接包集成 5 项
（准入无 fail、结论为 conditional、条数与类别、12 例无缺项）。交接包不存在时集成测试自动跳过，
`HANDOFF_DIR` 环境变量可以指向别的位置。

CI 只跑 `api/tests` 与 `scripts/test_export_catalog.py`，本文件的测试目前是本地跑；
如果要进 CI，需要把交接包挂成 artifact 或只跑纯函数部分。

## 4. 已知坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `UnicodeEncodeError: 'gbk' codec can't encode ...` | PowerShell 控制台默认 GBK，脚本里有 `⁻⁶` 之类字符 | 脚本已在 `main()` 里 `sys.stdout.reconfigure(encoding='utf-8')`，新脚本照抄这一行 |
| `FileExistsError: Database already exists` | `build_database` 不覆盖 | 换文件名，或先删旧库 |
| 图幅名报「CTX 网格中无 E-084_N-08」 | 拿 `ctx_grid` 的文件名当图幅名 | 从 shapefile 的 `Name` 字段读 |
| 案例纬度符号反了 | `native_window_world_m` 的 `y` 向南为正 | `lat = −y·180/(πR)` |
| `pip install` 卡住无输出 | pip 网络层异常 | 用 `--no-index --find-links` 离线装 |
| `npm`/`git` 相关命令报 TLS 或执行策略错误 | schannel 异常、PowerShell 脚本受限 | 见第 1 周 `docs/week1/run.md` |
