"""Check the atlas text against the catalogue, the public samples and the citations.

Week 4 asks for one table that answers, per case: is the catalogue record there,
is the image public, where does the per-pixel scale come from, is the citation
locatable, and what is still missing. The checks run against the private handoff
(never committed) plus the public sample list, so the table can be regenerated
instead of hand-maintained.

Usage:
    python scripts/check_content_assets.py --out output/week4_atlas_handoff
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONTENT = REPO / "mobile" / "src" / "content" / "atlas.json"
DEFAULT_SAMPLES = REPO / "examples" / "recognition.samples.json"
DEFAULT_SUBSET = REPO / "examples" / "catalog.real-subset.json"
DEFAULT_HANDOFF = Path(r"D:\python_project\teng_jiajie_week1_20261024")
TYPES = ("SL", "SP", "FL", "CL")
TYPE_NAMES = {"SL": "滑移型", "SP": "滑塌型", "FL": "流动型", "CL": "复合型"}
SCALE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m/像素")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_ids(subset_path: Path) -> set[str]:
    payload = read_json(subset_path)
    records = payload.get("records", payload)
    return {str(record.get("id")) for record in records if record.get("id")}


def samples_by_id(samples_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(samples_path)
    records = payload.get("records", payload)
    return {str(record["id"]): record for record in records}


def index_by_catalog_id(index_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["catalog_id"]): row for row in read_json(index_path)}


def window_metrics(row: dict[str, Any]) -> tuple[float, float]:
    """Return (window width in metres, metres per pixel) for the 224 px crop."""
    x0, _, x1, _ = (float(value) for value in row["native_window_world_m"])
    width = abs(x1 - x0)
    return width, width / 224.0


def scale_of(case: dict[str, Any]) -> float | None:
    match = SCALE_PATTERN.search(str(case.get("previewScale", "")))
    return float(match.group(1)) if match else None


def build_rows(
    content: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    index: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, Any]],
    public: set[str],
    sample_root: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    problems: list[str] = []
    for case in content["cases"]:
        case_id = case["id"]
        landslide_id = case["landslideId"]
        kind = case["type"]
        record = catalog.get(landslide_id)
        image = index.get(landslide_id)
        sample = samples.get(landslide_id)
        gaps: list[str] = []

        if record is None:
            problems.append(f"{case_id}：目录中没有 {landslide_id}")
            catalog_state = "未关联"
        elif record.get("type") != kind:
            problems.append(f"{case_id}：目录类别 {record.get('type')} 与正文 {kind} 不一致")
            catalog_state = f"类别不符（{record.get('type')}）"
        else:
            catalog_state = "已在公开子集" if landslide_id in public else "仅在完整目录"

        scale_rows = scale_of(case)
        if image is None:
            problems.append(f"{case_id}：没有 {landslide_id} 的裁片记录")
            scale_state = "无裁片记录"
            tile_state = "无"
        else:
            width, meters = window_metrics(image)
            scale_state = f"窗口 {width:.0f} m ÷ 224 px = {meters:.1f} m/像素"
            if scale_rows is not None and abs(round(meters, 1) - scale_rows) > 0.15:
                problems.append(
                    f"{case_id}：previewScale {scale_rows} 与裁片窗口算出的 {meters:.1f} 不一致"
                )
            case_tiles = image.get("source_tiles") or []
            if case_tiles and case["tile"] not in case_tiles:
                problems.append(f"{case_id}：图幅 {case['tile']} 不在裁片记录 {case_tiles} 里")
            tile_state = "、".join(case_tiles) or "未登记"

        if sample is None:
            image_state = "暂无公开影像"
            gaps.append("公开影像缺失 → 崔亚鹏（公开范围）/滕佳杰（申请资源）")
        else:
            path = sample_root / str(sample["imageUrl"]).lstrip("/")
            if not path.is_file():
                problems.append(f"{case_id}：公开影像文件不存在 {path}")
                image_state = "文件缺失"
                gaps.append("公开影像文件缺失 → 滕佳杰")
            else:
                digest = sha256_of(path)
                listed = str(sample.get("sha256", "")).strip().lower()
                packed = str((image or {}).get("image_sha256", "")).strip().lower()
                if not listed or not packed:
                    missing = "公开清单" if not listed else "交接包 image_index"
                    problems.append(f"{case_id}：公开影像 sha256 登记缺失（{missing} 未登记）")
                    image_state = "哈希登记缺失"
                    gaps.append(f"补 {missing} 的 sha256 → 滕佳杰")
                elif digest != listed or digest != packed:
                    differs = []
                    if digest != listed:
                        differs.append("公开清单")
                    if digest != packed:
                        differs.append("交接包 image_index")
                    problems.append(f"{case_id}：公开影像 sha256 与{'、'.join(differs)}不一致")
                    image_state = f"哈希不符（{'、'.join(differs)}）"
                    gaps.append("公开影像哈希不符 → 滕佳杰")
                elif sample.get("type") != kind:
                    problems.append(f"{case_id}：公开影像类型 {sample.get('type')} 与正文 {kind} 不一致")
                    image_state = "类型不符"
                    gaps.append("影像类型与正文不符 → 滕佳杰")
                else:
                    image_state = f"有（{path.name}，224×224，哈希双边一致：清单+交接包）"

        locator = str(case.get("citationLocator", ""))
        if not locator:
            problems.append(f"{case_id}：缺少引用定位")
            citation_state = "缺引用"
            gaps.append("引用定位缺失 → 滕佳杰")
        elif "页" not in locator:
            problems.append(f"{case_id}：引用定位没有页码：{locator}")
            citation_state = "无页码"
            gaps.append("引用补页码 → 滕佳杰")
        else:
            citation_state = "已核（PDF 物理页）"

        rows.append(
            {
                "case_id": case_id,
                "landslide_id": landslide_id,
                "type": f"{kind} {TYPE_NAMES.get(kind, '')}",
                "catalog": catalog_state,
                "image": image_state,
                "scale": scale_state,
                "tile": tile_state,
                "citation": citation_state,
                "gaps": "；".join(gaps) or "—",
            }
        )
    return rows, problems


def write_markdown(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    lines = [
        "# 图鉴案例资源核对表",
        "",
        f"内容版本 `{summary['version']}`；12 例：公开影像 {summary['withImage']} 例，"
        f"仅目录/仅文本 {12 - summary['withImage']} 例。本表由 `scripts/check_content_assets.py` 生成，"
        "可随着交接包重跑，不手工维护。",
        "",
        "| caseId | landslideId | 类型 | 目录关联状态 | 公开影像状态 | 尺度来源 | 图幅 | 引用状态 | 缺项/负责人 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {landslide_id} | {type} | {catalog} | {image} | {scale} | {tile} | "
            "{citation} | {gaps} |".format(**row)
        )
    lines += [
        "",
        "## 口径说明",
        "",
        "- **目录关联状态**：`已在公开子集` 指该记录列在 `examples/catalog.real-subset.json`；"
        "`仅在完整目录` 指记录存在于私有全库，但不在公开子集，公开仓库不复制其坐标。",
        "- **公开影像状态**：只核对 `examples/recognition.samples.json` 登记的 224×224 裁片，"
        "并重算 SHA-256；要求公开清单与交接包 `image_index.json` **两边同时匹配**才写「哈希双边一致」，"
        "只对上一方（或一边缺登记）都记为不符；未登记的一律写“暂无公开影像”，不用私有图片补位。",
        "- **尺度来源**：由裁片的 `native_window_world_m` 宽度除以 224 像素算出，"
        "与正文 `previewScale` 交叉核对；不同案例尺度不同，不能写成全批统一值。",
        "- **引用状态**：页码为 PDF 物理页码（含封面，共 28 页），与期刊印刷页码不同。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["case_id", "landslide_id", "type", "catalog", "image", "scale", "tile", "citation", "gaps"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    content_path: Path = DEFAULT_CONTENT,
    handoff: Path = DEFAULT_HANDOFF,
    samples_path: Path = DEFAULT_SAMPLES,
    subset_path: Path = DEFAULT_SUBSET,
    sample_root: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    content = read_json(content_path)
    catalog = {str(record["id"]): record for record in read_json(handoff / "01_catalog" / "mars_landslides.json")}
    index = index_by_catalog_id(handoff / "03_images" / "image_index.json")
    samples = samples_by_id(samples_path)
    public = public_ids(subset_path)
    root = sample_root or (REPO / "mobile" / "public")

    rows, problems = build_rows(content, catalog, index, samples, public, root)
    summary = {
        "version": content["version"],
        "cases": len(rows),
        "withImage": sum(1 for row in rows if row["image"].startswith("有")),
        "publicRecords": sum(1 for row in rows if row["catalog"] == "已在公开子集"),
        "byType": {kind: sum(1 for row in rows if row["type"].startswith(kind)) for kind in TYPES},
        "publicSamples": len(samples),
        "problems": problems,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_markdown(out / "atlas-asset-check.md", rows, summary)
        write_csv(out / "atlas-asset-check.csv", rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核对图鉴案例与目录、公开影像、引用")
    parser.add_argument("--content", type=Path, default=DEFAULT_CONTENT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--sample-root", type=Path, help="公开裁片所在目录，默认 mobile/public")
    parser.add_argument("--out", type=Path, help="把核对表写到这个目录")
    args = parser.parse_args(argv)
    if not (args.handoff / "01_catalog" / "mars_landslides.json").is_file():
        print(f"找不到交接包：{args.handoff}")
        return 2
    summary = run(
        content_path=args.content,
        handoff=args.handoff,
        samples_path=args.samples,
        subset_path=args.subset,
        sample_root=args.sample_root,
        out=args.out,
    )
    for problem in summary["problems"]:
        print(f"问题：{problem}")
    print(
        "核对完成：{cases} 例，公开影像 {withImage} 例，公开目录 {publicRecords} 例，"
        "问题 {count} 条".format(count=len(summary["problems"]), **summary)
    )
    return 0 if not summary["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
