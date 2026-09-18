# backend

Django + DRF。第 1 周已提供：滑坡表、列表/详情接口、CORS、演示种子数据。

## 启动

仓库根目录已有 `.venv`（Windows / Anaconda Python 3.13）。本机 3306 若被占用，默认走 SQLite，接口字段与 MySQL 方案相同。

```bash
# 仓库根目录
.venv/Scripts/activate          # Windows Git Bash
cd backend
python manage.py migrate
python manage.py seed_landslides
python manage.py runserver
```

验证：

- 列表 http://127.0.0.1:8000/api/landslides/
- 详情 http://127.0.0.1:8000/api/landslides/ls_0001/
- 筛选 `?type=slump&min_lat=-20&max_lat=0&min_lon=-90&max_lon=-40`

测试：`python manage.py test landslides`

## 切到 MySQL

1. 准备好库 `mars_kg`（Docker Compose 或本机实例）。
2. 根目录 `.env` 中设置 `DJANGO_DB_ENGINE=mysql` 以及 `MYSQL_*`。
3. 再跑 `migrate` 和 `seed_landslides`。

## 应用

- `landslides`：本周接口
- `graph_api` / `analytics` / `agent`：第 2–3 周挂图谱、预测推荐、RAG
