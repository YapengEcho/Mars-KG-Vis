# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

成都理工大学《Python程序设计》课程小组项目（自拟题）。主题：**火星滑坡分类知识图谱与可视化系统**，多终端（PC / 移动端响应式）、多模态（语音、手势）、专属 RAG 智能体。

课程硬性链路必须保留：**采集/爬取 → 清洗挖掘 → 数据库 → 终端可视化的预测与推荐**。图谱、RAG、语音、手势是加分，不能替代预测/推荐。

最终交付：源代码 + GitHub 链接、2 分钟演示视频、项目报告（需求分析、功能模块、加分项要写进报告）。

## 成员与目录归属

| 成员 | 职责 | 主要目录 |
| --- | --- | --- |
| 崔亚鹏 | 组长、Django/DRF、MySQL、Redis、Git/PR | `backend/` |
| 滕佳杰 | 采集清洗、特征、Neo4j、入库 | `data/` |
| 梁先楷 | Vue 大屏、PC/移动适配、语音/手势 | `frontend/` |
| 周璇 | 预测/推荐、RAG、测试、报告与视频脚本 | `ai/`、`docs/项目报告大纲.md` |

跨模块改动走 PR，不要直接提交 `main`。分支：`feat-backend`、`feat-frontend`、`feat-crawler`、`feat-ai`。

## 常用命令

仓库尚无完整骨架时，以各子目录 README 为准。约定启动方式：

```bash
# 基础设施
docker compose up -d mysql neo4j redis

# 后端（仓库根目录已有 .venv；默认 SQLite，.env 里 DJANGO_DB_ENGINE=mysql 可切库）
.venv/Scripts/activate          # Windows Git Bash
pip install -r backend/requirements.txt
cd backend
python manage.py migrate
python manage.py seed_landslides
python manage.py runserver

# 前端
cd frontend
npm install
npm run dev

# 数据入库（清洗脚本由 data/ 提供）
python -m data.scripts.ingest

# 构建向量库
python -m ai.rag.ingest

# 测试
cd backend && python manage.py test
cd ai && pytest
```

单测按 Django 应用跑：`python manage.py test landslides`。前端单测未定时不要编造测试命令。

## 架构

前后端分离：

- `frontend/`：Vue 3 + Vite + Element Plus。PC 可视化大屏为主，移动端同一套代码做响应式，不单独做原生 App。
- `backend/`：Django + DRF，唯一对外 HTTP 入口。前端、智能体、预测推荐都走这里，禁止前端直连 Neo4j / Chroma / DashScope。
- `data/`：原始数据、清洗脚本、图谱 schema、入 MySQL/Neo4j 的管道。
- `ai/`：sklearn 预测与相似推荐、LangChain RAG（Chroma + 通义千问）、RAG 评测脚本。

数据流：公开目录/爬虫 → pandas 清洗 → MySQL（查询与统计）+ Neo4j（关系推理）+ Chroma（文档切片）。Django 读三套存储，组 REST/JSON 给前端。

课程评分相关的技术选择（不要擅自替换）：

- 后端必须是 Django（老师加分示例是 Vue / Django / Redis / AI）。
- LLM 用通义千问（国产框架加分），不要默认改成 OpenAI。
- 向量库用 Chroma，不要上 Milvus。
- 预测用 scikit-learn，不要一上来上深度学习。
- 语音默认 Web Speech API；讯飞仅作备选。手势仅 PC 演示，MediaPipe Hands。

## 接口约定

实现或修改 API 时对齐 `docs/03-里程碑与接口约定.md`。当前冻结的资源：

- `GET /api/landslides/` 列表筛选
- `GET /api/landslides/{id}/` 详情
- `GET /api/stats/overview/` 大屏统计
- `GET /api/graph/subgraph/` 图谱子图
- `POST /api/agent/ask/` RAG 问答
- `POST /api/predict/` 类型预测
- `GET /api/recommend/` 相似滑坡推荐

字段用 snake_case。分页 `page` + `page_size`。错误体 `{ "detail": "..." }`。

## Git

规范原文见 `docs/04-Git协作规范.md`。强制：

- 禁止直接在 `main` 提交/推送；合入必须 PR，组长审核。
- 先 `git checkout main && git pull` 再 `git checkout -b feat-xxx`。
- Commit：`feat:` / `fix:` / `docs:` / `merge:`，禁止「更新代码」。
- 冲突在功能分支解决：`git merge main`，不要强制合入。

不要把 `.env`、原始数据 dump、Chroma 向量目录、模型权重提交进 Git。

## 演示约束

2 分钟视频必须能看到：地图/大屏、图谱、预测或推荐、智能体问答；语音或手势至少一条。Neo4j 若现场起不来，图谱接口需能回退到 `data/processed/graph_fallback.json`，保证演示不崩。
