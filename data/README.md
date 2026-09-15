# data

```
raw/          # 原始编目与爬取结果，全量不要提交
processed/    # landslides.csv、triples.csv、graph_fallback.json
scripts/      # download.py / crawl.py / clean.py / ingest_mysql.py / ingest_neo4j.py
graph/        # schema.cypher、实体说明
```

样本可以放 `raw/sample/`（几十条）方便别人没网也能跑通清洗。字段与图谱定义见 [docs/05-数据源与知识图谱草案.md](../docs/05-数据源与知识图谱草案.md)。
