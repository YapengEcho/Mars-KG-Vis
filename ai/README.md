# ai

周璇目录。建议：

```
models/train.py          # 读 data/processed/landslides.csv，导出 joblib
models/landslide_clf.joblib
rag/ingest.py            # 切片写入 ./chroma
rag/chain.py             # 检索 + 通义千问
eval/questions.json      # 5 个标准问题
```

Django 的 `predict` / `recommend` / `agent` 视图 import 这里的函数，不要复制一份逻辑到 backend。
