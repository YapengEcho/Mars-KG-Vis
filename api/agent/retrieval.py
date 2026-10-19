"""Simple keyword retriever for guidebook / case documents.
The Retriever exposes a single ``search`` method so the implementation can
later be swapped for vector retrieval without touching the agent loop.
"""
import json
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
def _count(haystack: str, terms: list[str]) -> int:
    low = haystack.casefold()
    return sum(low.count(t) for t in terms)
class KeywordRetriever:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = docs or []
    def search(self, query: str, doc_type: str | None = None, limit: int = 5) -> list[Doc]:
        terms = _query_terms(query)
        scored = []
        for doc in self.docs:
            if doc_type and doc.doc_type != doc_type:
                continue
            score = _count(doc.title, terms) * 3 + _count(doc.text, terms)
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
