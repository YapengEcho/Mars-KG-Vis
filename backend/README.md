# backend

Django + DRF。第 1 周由崔亚鹏初始化项目（建议项目名 `config`，应用 `landslides` / `graph_api` / `analytics` / `agent`）。

当前仓库只有约定，还没有 `manage.py`。初始化后把依赖写入 `requirements.txt`，并保证：

- 读仓库根目录 `.env`
- CORS 允许 `http://localhost:5173`
- 接口路径与 [docs/03-里程碑与接口约定.md](../docs/03-里程碑与接口约定.md) 一致
