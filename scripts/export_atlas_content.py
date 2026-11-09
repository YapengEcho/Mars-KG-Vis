"""Export the hand-curated atlas content as structured, searchable data.

The tracked source is ``mobile/src/content/atlas.json``: text, citation locators
and record ids only (no coordinates, areas or images), so the public repository
never carries a copy of the private catalogue. This script turns it into the
week-3 deliverables — a joined case table, a citation register and one JSONL
block per type/case/comparison for the Agent to retrieve — and, when a
catalogue is supplied, joins coordinates, area and class from it.

Usage:
    python scripts/export_atlas_content.py --out <dir> [--catalog mars_landslides.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONTENT = REPO / "mobile" / "src" / "content" / "atlas.json"
TYPES = ("SL", "SP", "FL", "CL")
TYPE_NAMES = {"SL": "滑移型", "SP": "滑塌型", "FL": "流动型", "CL": "复合型"}
REQUIRED_CASE_FIELDS = (
    "id",
    "landslideId",
    "type",
    "title",
    "subtitle",
    "region",
    "tile",
    "previewScale",
    "observations",
    "tags",
    "citationLocator",
)


class ContentError(ValueError):
    """Raised when the atlas content file breaks the contract."""


def load_content(path: Path) -> dict[str, Any]:
    content = json.loads(path.read_text(encoding="utf-8"))
    problems = check_content(content)
    if problems:
        raise ContentError("；".join(problems))
    return content


def check_content(content: dict[str, Any]) -> list[str]:
    """Validate the shape the APP and the Agent both rely on."""
    problems: list[str] = []
    if not isinstance(content, dict):
        return ["图鉴内容不是对象"]
    if not isinstance(content.get("version"), str):
        problems.append("缺少版本号")
    source = content.get("sourceDocument")
    if not isinstance(source, dict) or not str(source.get("url", "")).startswith("https://"):
        problems.append("缺少来源文献或文献地址不是 HTTPS")
    types = content.get("types")
    if not isinstance(types, list) or len(types) != len(TYPES):
        problems.append("类型数量不是 4")
        types = []
    seen_types: set[str] = set()
    registered: dict[str, str] = {}
    for entry in types:
        if not isinstance(entry, dict):
            problems.append("类型条目不是对象")
            continue
        kind = entry.get("type")
        if kind not in TYPES:
            problems.append(f"未知类型：{kind}")
            continue
        if kind in seen_types:
            problems.append(f"类型重复：{kind}")
        seen_types.add(kind)
        for field in ("quote", "deposits", "flow", "confusable", "recognize", "citations"):
            if not entry.get(field):
                problems.append(f"{kind} 缺少 {field}")
        for case_id in entry.get("caseIds") or []:
            if case_id in registered:
                problems.append(f"案例 {case_id} 重复登记")
            registered[case_id] = kind
    cases = content.get("cases")
    if not isinstance(cases, list) or not cases:
        problems.append("缺少案例")
        cases = []
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            problems.append("案例条目不是对象")
            continue
        case_id = str(case.get("id", ""))
        if case_id in seen_cases:
            problems.append(f"案例编号重复：{case_id}")
        seen_cases.add(case_id)
        for field in REQUIRED_CASE_FIELDS:
            if not case.get(field):
                problems.append(f"{case_id} 缺少 {field}")
        if case.get("type") not in TYPES:
            problems.append(f"{case_id} 类型无效")
        elif not str(case.get("landslideId", "")).startswith("ls_"):
            problems.append(f"{case_id} 的目录编号无效")
        elif registered.get(case_id) != case.get("type"):
            problems.append(f"{case_id} 未登记在 {case.get('type')} 的案例清单里")
    comparisons = content.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) < 2:
        problems.append("至少需要两组类型对比")
        comparisons = []
    for item in comparisons:
        if not isinstance(item, dict):
            problems.append("对比条目不是对象")
            continue
        if item.get("left") == item.get("right"):
            problems.append(f"对比 {item.get('id')} 两侧类型相同")
        if not item.get("rows"):
            problems.append(f"对比 {item.get('id')} 缺少对比项")
    if not content.get("limits"):
        problems.append("缺少判读限制")
    return problems


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        by_id[str(record.get("id", ""))] = record
    return by_id


def case_rows(content: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in content["cases"]:
        record = catalog.get(case["landslideId"], {})
        rows.append(
            {
                "case_id": case["id"],
                "landslide_id": case["landslideId"],
                "type": case["type"],
                "type_name": TYPE_NAMES.get(case["type"], ""),
                "title": case["title"],
                "subtitle": case["subtitle"],
                "region": case["region"],
                "tile": case["tile"],
                "preview_scale": case["previewScale"],
                "tags": "；".join(case["tags"]),
                "citation_locator": case["citationLocator"],
                "longitude": record.get("lon", ""),
                "latitude": record.get("lat", ""),
                "area_km2": record.get("area", ""),
                "in_catalog": "yes" if record else "no",
            }
        )
    return rows


def citation_rows(content: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in content["types"]:
        for citation in entry["citations"]:
            rows.append(
                {
                    "owner_kind": "type",
                    "owner_id": entry["type"],
                    "document_id": citation["documentId"],
                    "locator": citation["locator"],
                }
            )
    for item in content["comparisons"]:
        for citation in item["citations"]:
            rows.append(
                {
                    "owner_kind": "comparison",
                    "owner_id": item["id"],
                    "document_id": citation["documentId"],
                    "locator": citation["locator"],
                }
            )
    for case in content["cases"]:
        rows.append(
            {
                "owner_kind": "case",
                "owner_id": case["id"],
                "document_id": content["sourceDocument"]["id"],
                "locator": case["citationLocator"],
            }
        )
    return rows


def search_blocks(content: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for entry in content["types"]:
        text = "\n".join(
            [
                f"类型 {entry['type']} {entry['name']}（{entry['english']}）",
                f"定义：{entry['quote']}",
                f"堆积特征：{entry['deposits']}",
                f"流动特征：{entry['flow']}",
                f"识别要点：{'；'.join(entry['recognize'])}",
                f"易混类型：{entry['confusable']}",
                f"引用：{'；'.join(c['locator'] for c in entry['citations'])}",
            ]
        )
        blocks.append(
            {"id": f"type-{entry['type']}", "kind": "type", "title": f"{entry['type']} {entry['name']}", "text": text}
        )
    for case in content["cases"]:
        text = "\n".join(
            [
                f"案例 {case['id']}（{case['type']} {TYPE_NAMES.get(case['type'], '')}）{case['title']}",
                f"副标题：{case['subtitle']}",
                f"{case['region']} · {case['tile']} · {case['previewScale']}",
                f"观察要点：{'；'.join(case['observations'])}",
                f"标签：{'；'.join(case['tags'])}",
                f"引用：{case['citationLocator']}",
            ]
        )
        blocks.append({"id": case["id"], "kind": "case", "title": case["title"], "text": text})
    for item in content["comparisons"]:
        lines = [f"类型对比 {item['title']}", item["why"]]
        lines += [
            f"{row['dimension']}：{item['left']} {row['left']} / {item['right']} {row['right']}"
            for row in item["rows"]
        ]
        blocks.append({"id": f"compare-{item['id']}", "kind": "comparison", "title": item["title"], "text": "\n".join(lines)})
    return blocks


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(content_path: Path, out: Path, catalog_path: Path | None) -> dict[str, Any]:
    content = load_content(content_path)
    catalog = load_catalog(catalog_path) if catalog_path else {}
    out.mkdir(parents=True, exist_ok=True)

    cases = case_rows(content, catalog)
    citations = citation_rows(content)
    blocks = search_blocks(content)
    joined = {
        **content,
        "cases": [
            {**case, "record": catalog.get(case["landslideId"])} for case in content["cases"]
        ],
    }

    (out / "atlas-content.json").write_text(
        json.dumps(joined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        out / "atlas-cases.csv",
        cases,
        [
            "case_id",
            "landslide_id",
            "type",
            "type_name",
            "title",
            "subtitle",
            "region",
            "tile",
            "preview_scale",
            "tags",
            "citation_locator",
            "longitude",
            "latitude",
            "area_km2",
            "in_catalog",
        ],
    )
    write_csv(out / "atlas-citations.csv", citations, ["owner_kind", "owner_id", "document_id", "locator"])
    with (out / "atlas-search.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for block in blocks:
            handle.write(json.dumps(block, ensure_ascii=False) + "\n")
    (out / "atlas-limits.md").write_text(
        "# 判读限制\n\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(content["limits"], 1)) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": content["version"],
        "types": len(content["types"]),
        "cases": len(content["cases"]),
        "casesInCatalog": sum(1 for row in cases if row["in_catalog"] == "yes"),
        "comparisons": len(content["comparisons"]),
        "limits": len(content["limits"]),
        "citations": len(citations),
        "searchBlocks": len(blocks),
        "byType": {
            kind: sum(1 for case in content["cases"] if case["type"] == kind) for kind in TYPES
        },
    }
    (out / "atlas-export-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导图鉴内容为结构化数据与可检索文本")
    parser.add_argument("--content", type=Path, default=DEFAULT_CONTENT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--catalog", type=Path, help="目录 JSON（按 landslideId 关联坐标、面积与类别）")
    parser.add_argument("--check", action="store_true", help="只校验内容文件，不写文件")
    args = parser.parse_args(argv)

    if args.check or args.out is None:
        problems = check_content(json.loads(args.content.read_text(encoding="utf-8")))
        for problem in problems:
            print(f"问题：{problem}")
        print("图鉴内容校验通过" if not problems else f"图鉴内容 {len(problems)} 个问题")
        return 0 if not problems else 1

    summary = export(args.content, args.out, args.catalog)
    print(
        "图鉴导出：{cases} 条案例（目录中 {casesInCatalog} 条），{types} 类，{comparisons} 组对比，"
        "{citations} 条引用，{searchBlocks} 个检索片段 → {out}".format(out=args.out, **summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

