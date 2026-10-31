"""Simple keyword retriever for guidebook / case documents.
The Retriever exposes a single ``search`` method so the implementation can
later be swapped for vector retrieval without touching the agent loop.

Scoring is IDF-weighted: a term that appears in many documents (such as 滑坡)
contributes little, while a rare term (such as 编目 or 体积效应) dominates. Both
the title bonus and the body counts use the weight, and body counts grow
logarithmically so long documents cannot win by sheer repetition.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
@dataclass
class Doc:
    doc_id: str
    doc_type: str
    title: str
    text: str
    slide_ids: list[str] = field(default_factory=list)
    source_title: str = ""
    source_url: str | None = None
def _query_terms(query: str) -> list[str]:
    q = query.strip().casefold()
    terms = {q}
    terms.update(t for t in q.split() if t)
    # CJK character bigrams give sub-word matching without a segmentation library.
    terms.update(q[i : i + 2] for i in range(max(0, len(q) - 1)) if q[i : i + 2].strip())
    return [t for t in terms if t]
class KeywordRetriever:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = docs or []
        self._df: dict[str, int] | None = None
    def _build_df(self) -> None:
        self._df = {}
        for doc in self.docs:
            for term in set(_query_terms(doc.title + " " + doc.text)):
                self._df[term] = self._df.get(term, 0) + 1
    def _idf(self, term: str) -> float:
        if self._df is None:
            self._build_df()
        total = len(self.docs) or 1
        df = self._df.get(term, 0)
        return math.log((total + 1) / (df + 1)) + 1.0
    def search(self, query: str, doc_type: str | None = None, limit: int = 5) -> list[Doc]:
        terms = set(_query_terms(query))
        scored = []
        for doc in self.docs:
            if doc_type and doc.doc_type != doc_type:
                continue
            title = doc.title.casefold()
            text = doc.text.casefold()
            score = 0.0
            for term in terms:
                weight = self._idf(term)
                if term in title:
                    score += 3.0 * weight
                occurrences = text.count(term)
                if occurrences:
                    score += weight * (1.0 + math.log(occurrences))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:limit]]
def load_docs(path: str | Path) -> list[Doc]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    docs = []
    for item in raw.get("docs", []):
        docs.append(
            Doc(
                doc_id=item["doc_id"],
                doc_type=item["doc_type"],
                title=item["title"],
                text=item.get("text", ""),
                slide_ids=item.get("slide_ids", []),
                source_title=item.get("source_title", ""),
                source_url=item.get("source_url"),
            )
        )
    return docs
