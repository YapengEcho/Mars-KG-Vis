# 火星滑坡分类知识图谱与可视化系统

成都理工大学《Python程序设计》课程项目。前后端分离，面向火星滑坡编目数据：采集清洗、知识图谱、可视化大屏、类型预测与相似推荐、RAG 智能体，并在 PC / 移动端提供语音与手势交互。


## 成员

| 成员 | 角色 | 负责 |
| --- | --- | --- |
| 崔亚鹏 | 组长 / 后端 | 进度、Git/PR、Django 接口、MySQL、Redis |
| 滕佳杰 | 数据与图谱 | 采集清洗、特征、Neo4j、双库入库 |
| 梁先楷 | 前端与交互 | Vue 大屏、PC/移动适配、语音、手势 |
| 周璇 | 算法与交付 | 预测/推荐、RAG、测试、报告、视频脚本 |

## 技术栈（已冻结）

| 层 | 选型 |
| --- | --- |
| 前端 | Vue 3 + Vite + Element Plus + ECharts + Leaflet（火星底图）+ Three.js（首页星球） |
| 移动端 | 同一套 Vue 响应式，不做原生 App |
| 后端 | Django + Django REST Framework |
| 业务库 | MySQL 8 |
| 图数据库 | Neo4j Community（Docker） |
| 向量库 | Chroma（本地目录） |
| RAG | LangChain + 通义千问 |
| 预测/推荐 | scikit-learn |
| 缓存 | Redis |
| 语音 | Web Speech API（讯飞为备选） |
| 手势 | MediaPipe Hands（仅 PC） |
| 协作 | GitHub + PR |

决策理由与砍掉的范围见 [docs/01-评估与技术决策.md](docs/01-评估与技术决策.md)。

## 仓库结构

```
backend/     Django API
frontend/    Vue 大屏
data/        采集、清洗、图谱 schema
ai/          预测、推荐、RAG
docs/        立项、架构、接口、Git、报告大纲
```

## 课程要求对齐

必须能演示完整链路：

1. 互联网获取数据（公开目录下载 + 至少一条真实爬虫）
2. 清洗、处理、挖掘
3. 数据库存储（MySQL + Neo4j）
4. 终端可视化，并且包含**预测**和**推荐**

加分项：Vue / Django / Redis / AI、国产模型（通义千问）、Git、代码优化、AI 协助开发、AI 功能模块。这些都要写进项目报告。

## 文档

- [分工](分工.md)
- [评估与技术决策](docs/01-评估与技术决策.md)
- [系统架构](docs/02-系统架构.md)
- [里程碑与接口](docs/03-里程碑与接口约定.md)
- [Git 协作规范](docs/04-Git协作规范.md)
- [数据源与图谱草案](docs/05-数据源与知识图谱草案.md)
- [项目报告大纲](docs/06-项目报告大纲.md)

## 本地启动（骨架落地后）

```bash
docker compose up -d mysql neo4j redis
cd backend && python manage.py runserver
cd frontend && npm run dev
```

密钥放 `.env`，不要提交。演示前确认：大屏、图谱、预测或推荐、智能体问答、至少一条语音或手势指令。
