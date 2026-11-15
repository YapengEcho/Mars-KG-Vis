"""Convert the atlas content (mobile/src/content/atlas.json) into the
retrieval index consumed by api/agent/retrieval.py.

The atlas is the maintained source of the type/comparison/case text; the
guidebook JSON is the retrieval index built from it. This script merges, it
does not overwrite: entries whose doc_id is produced from the atlas are
replaced in place, entries that only exist in the index (distribution,
mobility, history, research method) are kept untouched.

Mapping (agreed in agent-recognition.md):
    atlas types        -> type_intro   (doc_id kept stable: doc-sl, doc-sp, ...)
    atlas comparisons  -> compare      (doc-compare-sl-sp, doc-compare-sp-cl)
    atlas cases        -> case         (doc-case-<landslideId>, new doc_type)
    atlas limits       -> method       (doc-limit)

Usage:
    python scripts/import_atlas_to_guidebook.py            # write the index
    python scripts/import_atlas_to_guidebook.py --check     # report only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_PATH = REPO_ROOT / "mobile" / "src" / "content" / "atlas.json"
INDEX_PATH = REPO_ROOT / "api" / "data" / "docs" / "guidebook.demo.json"

TYPE_DOC_IDS = {"SL": "doc-sl", "SP": "doc-sp", "FL": "doc-fl", "CL": "doc-cl"}
COMPARISON_DOC_IDS = {"sl-vs-sp": "doc-compare-sl-sp", "sp-vs-cl": "doc-compare-sp-cl"}
LIMITS_DOC_ID = "doc-limit"


def _locators(citations: list[dict]) -> str:
    return "；".join(item.get("locator", "") for item in citations if item.get("locator"))


def _source_title(document: dict, locator: str) -> str:
    title = document.get("title", "").strip()
    return f"《{title}》{locator}" if locator else f"《{title}》"


def _source_url(document: dict) -> str | None:
    url = (document.get("url") or "").strip()
    return url or None


def type_title(item: dict) -> str:
    """Build a searchable title: the atlas only stores "滑移型", but users ask
    "滑移型滑坡是什么", so the title carries the full name and the English term.
    """
    title = f"{item.get('type', '')} {item.get('name', '')}滑坡".strip()
    english = (item.get("english") or "").strip()
    return f"{title}（{english}）" if english else title


def type_doc(item: dict, document: dict) -> dict:
    lines = [item.get("quote", "").strip(), ""]
    if item.get("deposits"):
        lines.append(f"后运动堆积：{item['deposits']}")
    if item.get("flow"):
        lines.append(f"流动特征：{item['flow']}")
    if item.get("recognize"):
        lines.append("判读要点：")
        lines += [
            f"{index}. {point}"
            for index, point in enumerate(item["recognize"], start=1)
        ]
    if item.get("confusable"):
        lines.append(f"易混：{item['confusable']}")
    locator = _locators(item.get("citations", []))
    if locator:
        lines += ["", f"出处：{locator}"]
    return {
        "doc_id": TYPE_DOC_IDS.get(item.get("type", ""), f"doc-{item.get('type', '').lower()}"),
        "doc_type": "type_intro",
        "title": type_title(item),
        "text": "\n".join(line for line in lines if line is not None).strip(),
        "slide_ids": [],
        "source_title": _source_title(document, locator),
        "source_url": _source_url(document),
    }


def comparison_doc(item: dict, document: dict) -> dict:
    rows = item.get("rows", [])
    left_code = item.get("left", "")
    right_code = item.get("right", "")
    lines = [item.get("why", "").strip(), ""]
    lines.append(f"对比维度（左 {left_code} / 右 {right_code}）：")
    for row in rows:
        lines.append(f"- {row.get('dimension', '')}：左 {row.get('left', '')}；右 {row.get('right', '')}")
    locator = _locators(item.get("citations", []))
    if locator:
        lines += ["", f"出处：{locator}"]
    return {
        "doc_id": COMPARISON_DOC_IDS.get(item.get("id", ""), f"doc-compare-{item.get('id', '')}"),
        "doc_type": "compare",
        "title": item.get("title", ""),
        "text": "\n".join(lines).strip(),
        "slide_ids": [],
        "source_title": _source_title(document, locator),
        "source_url": _source_url(document),
    }


def case_doc(item: dict, document: dict) -> dict:
    lines = []
    if item.get("subtitle"):
        lines.append(item["subtitle"])
    meta = []
    for label, key in (("区域", "region"), ("CTX 图幅", "tile"), ("预览尺度", "previewScale")):
        if item.get(key):
            meta.append(f"{label}：{item[key]}")
    if meta:
        lines.append("；".join(meta))
    observations = item.get("observations", [])
    if observations:
        lines.append("观察要点：")
        lines += [f"- {point}" for point in observations]
    if item.get("tags"):
        lines.append("标签：" + "、".join(item["tags"]))
    locator = (item.get("citationLocator") or "").strip()
    if locator:
        lines += ["", f"出处：{locator}"]
    slide_id = (item.get("landslideId") or "").strip()
    return {
        "doc_id": f"doc-case-{slide_id}" if slide_id else f"doc-{item.get('id', '')}",
        "doc_type": "case",
        "title": f"案例：{item.get('title', '')}（{item.get('type', '')}，{slide_id}）",
        "text": "\n".join(lines).strip(),
        "slide_ids": [slide_id] if slide_id else [],
        "source_title": _source_title(document, locator),
        "source_url": _source_url(document),
    }


def limits_doc(limits: list[str], document: dict, existing: dict | None) -> dict:
    text = "\n".join(f"- {line}" for line in limits).strip()
    locator = "3 节判读边界" if limits else ""
    return {
        "doc_id": LIMITS_DOC_ID,
        "doc_type": "method",
        "title": "使用本编目时需注意的限制",
        "text": text,
        "slide_ids": [],
        "source_title": _source_title(document, locator),
        "source_url": _source_url(document),
    }


def build(atlas: dict, existing_docs: list[dict]) -> list[dict]:
    document = atlas.get("sourceDocument", {})
    by_id = {doc.get("doc_id"): doc for doc in existing_docs}

    for item in atlas.get("types", []):
        doc = type_doc(item, document)
        by_id[doc["doc_id"]] = doc
    for item in atlas.get("comparisons", []):
        doc = comparison_doc(item, document)
        by_id[doc["doc_id"]] = doc
    for item in atlas.get("cases", []):
        doc = case_doc(item, document)
        by_id[doc["doc_id"]] = doc
    if atlas.get("limits"):
        doc = limits_doc(atlas["limits"], document, by_id.get(LIMITS_DOC_ID))
        by_id[doc["doc_id"]] = doc

    # Keep the original order for docs that already existed, then append new ones.
    ordered: list[dict] = []
    seen: set[str] = set()
    for doc in existing_docs:
        doc_id = doc.get("doc_id")
        if doc_id in by_id and doc_id not in seen:
            ordered.append(by_id[doc_id])
            seen.add(doc_id)
    for doc_id, doc in by_id.items():
        if doc_id not in seen:
            ordered.append(doc)
            seen.add(doc_id)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    atlas = json.loads(ATLAS_PATH.read_text(encoding="utf-8-sig"))
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8-sig"))
    existing_docs = index.get("docs", [])

    docs = build(atlas, existing_docs)
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc["doc_type"]] = counts.get(doc["doc_type"], 0) + 1

    print(f"atlas version   : {atlas.get('version')}")
    print(f"atlas entries   : types={len(atlas.get('types', []))} "
          f"comparisons={len(atlas.get('comparisons', []))} "
          f"cases={len(atlas.get('cases', []))} limits={len(atlas.get('limits', []))}")
    print(f"index before    : {len(existing_docs)} docs")
    print(f"index after     : {len(docs)} docs")
    print("doc_type counts : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    added = [d["doc_id"] for d in docs if d["doc_id"] not in {x.get("doc_id") for x in existing_docs}]
    print(f"added           : {len(added)} -> {added}")

    if args.check:
        return 0

    payload = {
        "atlasVersion": atlas.get("version"),
        "generatedBy": "scripts/import_atlas_to_guidebook.py",
        "docs": docs,
    }
    INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"written         : {INDEX_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
