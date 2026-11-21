"""Generate the week-5 demo figures: demo CTX canvases, charts and page shots.

Three kinds of output, all written under ``--out`` (never into the repository,
which stays free of image assets):

* ``demo-ctx-0N-*.png``      upload-ready demo canvases (≤2048 px, PNG, ≤8 MiB)
* ``fig-demo-input.png``     one sheet that explains the three inputs
* ``chart-types.png``        catalogue vs case counts per type
* ``chart-scale.png``        per-case capture-window scale
* ``shot-*.png``             phone screenshots (needs ``--base-url``)

Usage:
    python scripts/capture_demo_figures.py --out output/week5 \\
        --base-url http://127.0.0.1:5174

Requires Pillow. Screenshots additionally start a headless browser through
``scripts/cdp_screenshot.py``. Exit codes: 0 ok, 1 a figure failed its checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "mobile" / "public" / "samples"
CONTENT = REPO / "mobile" / "src" / "content" / "atlas.json"
SUBSET = REPO / "examples" / "catalog.real-subset.json"
SAMPLE_INDEX = REPO / "examples" / "recognition.samples.json"
FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BG = (11, 16, 23)
TEXT = (241, 244, 249)
MUTED = (155, 169, 189)
LINE = (42, 55, 74)
TYPE_COLORS = {
    "SL": (236, 156, 121),
    "SP": (120, 190, 212),
    "FL": (185, 162, 232),
    "CL": (225, 197, 116),
}
TYPE_NAMES = {"SL": "滑移型", "SP": "滑塌型", "FL": "流动型", "CL": "复合型"}

# Demo inputs: public previews only, one per easy / mixed / small case.
DEMO_CASES = (
    ("ls_0493", "demo-ctx-01-ls0493.png", "SL"),
    ("ls_0528", "demo-ctx-02-ls0528.png", "CL"),
    ("ls_0597", "demo-ctx-03-ls0597.png", "FL"),
)
UPSCALE = 8
PREVIEW_SIDE = 224
MAX_EDGE = 2048
MAX_BYTES = 8 * 1024 * 1024


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        if draw.textlength(current + char, font=fnt) <= width:
            current += char
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_ids() -> set[int]:
    payload = json.loads(SUBSET.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return {int(item["sourceId"]) for item in records}


def scale_of(case: dict) -> float:
    return float(str(case["previewScale"]).split()[0])


def sample_index(path: Path = SAMPLE_INDEX) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return {str(item["id"]): item for item in records}


def verify_preview(source: Path, sample: dict) -> list[str]:
    """Check a public preview against the published sample list before using it."""
    problems: list[str] = []
    sid = sample["id"]
    digest = sha256_of(source)
    if digest != sample["sha256"]:
        problems.append(
            f"{sid}: 源图 SHA-256 与公开样例清单不一致"
            f"（清单 {sample['sha256'][:12]}…，实际 {digest[:12]}…），疑似同名图混入，本次不生成"
        )
    with Image.open(source) as image:
        if [image.width, image.height] != [sample["width"], sample["height"]]:
            problems.append(
                f"{sid}: 源图尺寸 {image.width}×{image.height} 与清单 "
                f"{sample['width']}×{sample['height']} 不一致"
            )
    return problems


def replacement_of(name: str, out: Path, native_dir: Path | None, previous: dict[str, dict]) -> tuple[Path, str] | None:
    """A native crop wins over the upscaled preview; a hand-placed file stays hand-placed.

    Two signals count as "not ours": the previous manifest recorded a native/replacement
    source, or the file's hash differs from what the manifest recorded for a preview. The
    first one matters on the second and later reruns, where the hash does match.
    """
    if native_dir is not None:
        candidate = native_dir / name
        if candidate.exists():
            return candidate, "native"
    target = out / name
    if not target.exists():
        return None
    recorded = previous.get(name, {})
    if recorded.get("sourceMode") in ("native", "replacement"):
        return target, str(recorded["sourceMode"])
    if recorded.get("sha256") != sha256_of(target):
        return target, "replacement"
    return None


def describe_native(path: Path, case: dict, mode: str) -> dict:
    window_m = scale_of(case) * PREVIEW_SIDE
    with Image.open(path) as image:
        size = [image.width, image.height]
    return {
        "file": path.name,
        "source": path.name,
        "sourceSize": size,
        "sourceSha256": sha256_of(path),
        "sourceMode": mode,
        "upscale": None,
        "size": size,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "windowM": round(window_m, 2),
        "fileScale": round(window_m / max(size[0], 1), 3),
    }


def demo_canvas(source: Path, out: Path, case: dict) -> dict:
    image = Image.open(source).convert("L")
    size = image.width * UPSCALE
    if size > MAX_EDGE:
        raise ValueError(f"{source.name}: 放大后 {size} 超过 {MAX_EDGE} 像素上限")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), Image.Resampling.LANCZOS).save(out, optimize=True)
    window_m = scale_of(case) * PREVIEW_SIDE
    return {
        "file": out.name,
        "source": str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source),
        "sourceSize": list(image.size),
        "sourceSha256": sha256_of(source),
        "sourceMode": "preview",
        "upscale": UPSCALE,
        "size": [size, size],
        "bytes": out.stat().st_size,
        "sha256": sha256_of(out),
        "windowM": round(window_m, 2),
        "fileScale": round(window_m / size, 3),
    }


def demo_sheet(inputs: list[dict], out: Path) -> None:
    width, height = 1240, 1000
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((56, 44), "第 5 周演示输入：三张 CTX 裁图", font=font(34, True), fill=TEXT)
    upscaled = [meta for meta in inputs if meta.get("sourceMode") == "preview"]
    native = [meta for meta in inputs if meta.get("sourceMode") != "preview"]
    if not native:
        header = (
            f"每张都是 {PREVIEW_SIDE}×{PREVIEW_SIDE} 公开预览按 {UPSCALE} 倍放大到 "
            f"{PREVIEW_SIDE * UPSCALE}×{PREVIEW_SIDE * UPSCALE}，满足上传上限，便于滑动窗口检出多个候选"
        )
    else:
        header = (
            f"{len(upscaled)} 张是公开预览放大 {UPSCALE} 倍、{len(native)} 张是原生裁图替换；"
            f"尺寸与体积均满足上传上限（最长边 ≤{MAX_EDGE}、≤8 MiB）"
        )
    draw.text((56, 96), header, font=font(19), fill=MUTED)
    thumb, left = 260, 56
    for index, meta in enumerate(inputs):
        x = left + index * (thumb + 40)
        y = 168
        preview = Image.open(meta["path"]).convert("L").resize((thumb, thumb), Image.Resampling.LANCZOS)
        image.paste(Image.merge("RGB", (preview, preview, preview)), (x, y))
        draw.rectangle([x, y, x + thumb, y + thumb], outline=LINE, width=2)
        draw.text((x, y + thumb + 16), meta["sid"], font=font(21, True), fill=TEXT)
        draw.text(
            (x, y + thumb + 46),
            TYPE_NAMES[meta["type"]],
            font=font(17),
            fill=TYPE_COLORS[meta["type"]],
        )
        caption_y = y + thumb + 72
        for line in wrap(draw, meta["subtitle"], font(15), thumb):
            draw.text((x, caption_y), line, font=font(15), fill=MUTED)
            caption_y += 24
        if meta.get("sourceMode") == "preview":
            scale_line = f"原窗口 {meta['sourceScale']:.1f} m/像素 · 放大 {meta['upscale']}×"
        else:
            mode = "原生裁图" if meta.get("sourceMode") == "native" else "替换裁片"
            scale_line = f"{mode} {meta['fileScale']:.1f} m/像素 · 窗口不变"
        draw.text(
            (x, caption_y + 4),
            f"{meta['size'][0]}×{meta['size'][1]}",
            font=font(15),
            fill=MUTED,
        )
        caption_y += 26
        for line in wrap(draw, scale_line, font(15), thumb):
            draw.text((x, caption_y + 4), line, font=font(15), fill=MUTED)
            caption_y += 24
    notes = [
        "用法：在移动端「识别」页选择其中一张上传，等候选框，再点开候选看四类分数与待复核标记。",
        "边界：放大不增加真实分辨率，等效分辨率变成原窗口的 1/8；分数只用于联调，不能当精度结论。",
        "来源：裁片取自交接包 03_images/train224 的公开 8 例预览，生成前按公开样例清单核对 SHA-256，对不上直接中止。",
        "替换：原生 6 m CTX 裁图按同名放进 --out 或 --native-dir 后重跑，脚本保留原生裁片、不再放大，并刷新这张图与清单。",
    ]
    y = 600
    draw.text((56, y), "怎么用、怎么讲、边界在哪", font=font(24, True), fill=TEXT)
    y += 48
    for note in notes:
        for line in wrap(draw, "· " + note, font(19), width - 112):
            draw.text((56, y), line, font=font(19), fill=MUTED)
            y += 32
        y += 8
    image.save(out, optimize=True)


def chart_types(content: dict, catalog_total: int, out: Path) -> None:
    counts = {key: len([c for c in content["cases"] if c["type"] == key]) for key in TYPE_COLORS}
    catalogue = {"SL": 1787, "SP": 321, "FL": 50, "CL": 813}
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((64, 48), f"编目 {catalog_total} 条与 12 个案例的四类分布", font=font(34, True), fill=TEXT)
    draw.text((64, 100), "条形为编目记录数；下方小字为本周案例数（每类 3 例）", font=font(20), fill=MUTED)
    chart_top, chart_bottom, bar_width, gap = 190, 580, 150, 90
    x = 96
    for key in ("SL", "SP", "FL", "CL"):
        value = catalogue[key]
        bar_height = int((chart_bottom - chart_top) * value / max(catalogue.values()))
        top = chart_bottom - bar_height
        draw.rectangle([x, top, x + bar_width, chart_bottom], fill=TYPE_COLORS[key])
        draw.text((x, top - 44), f"{value}", font=font(26, True), fill=TEXT)
        draw.text((x, chart_bottom + 18), f"{TYPE_NAMES[key]} {key}", font=font(22, True), fill=TEXT)
        draw.text(
            (x, chart_bottom + 52),
            f"{counts[key]} 例（{value / catalog_total * 100:.1f}%）",
            font=font(18),
            fill=MUTED,
        )
        x += bar_width + gap
    draw.line([64, chart_bottom, width - 64, chart_bottom], fill=LINE, width=2)
    draw.text(
        (64, height - 56),
        "来源：老师提供的火星滑坡编目（2971 条）与 mobile/src/content/atlas.json",
        font=font(18),
        fill=MUTED,
    )
    image.save(out, optimize=True)


def chart_scale(content: dict, out: Path) -> None:
    cases = sorted(content["cases"], key=scale_of)
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((64, 48), "12 个案例的取图窗口尺度（米/像素）", font=font(34, True), fill=TEXT)
    draw.text(
        (64, 100),
        f"窗口为人工框长边的 1.5 倍；最小 {scale_of(cases[0]):.1f}、最大 {scale_of(cases[-1]):.1f}，相差约 10 倍",
        font=font(20),
        fill=MUTED,
    )
    left, right, top, row_height, axis_max = 300, width - 80, 180, 42, 210
    bottom = top + row_height * len(cases)
    for tick in range(0, axis_max + 1, 30):
        x = left + int((right - left) * tick / axis_max)
        draw.line([x, top - 12, x, bottom + 8], fill=LINE, width=1)
        draw.text((x - 16, bottom + 16), str(tick), font=font(16), fill=MUTED)
    for index, case in enumerate(cases):
        y = top + index * row_height
        value = scale_of(case)
        bar_right = left + int((right - left) * value / axis_max)
        draw.rectangle([left, y + 9, bar_right, y + 30], fill=TYPE_COLORS[case["type"]])
        draw.text((64, y + 9), case["landslideId"], font=font(18), fill=TEXT)
        draw.text((bar_right + 12, y + 9), f"{value:.1f}", font=font(18, True), fill=TEXT)
    draw.text(
        (64, height - 44),
        "尺度取自交接包 image_index.json 的 native_window_world_m，逐张不同，没有统一值",
        font=font(18),
        fill=MUTED,
    )
    image.save(out, optimize=True)


def describe(path: Path) -> dict:
    with Image.open(path) as image:
        size = list(image.size)
    return {
        "file": path.name,
        "size": size,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def screenshots(
    base_url: str, out_dir: Path, cdp: Path, routes: dict[str, str], problems: list[str]
) -> list[dict]:
    rows = []
    for name, route in routes.items():
        target = out_dir / name
        result = subprocess.run(
            [
                sys.executable,
                str(cdp),
                "--url",
                f"{base_url.rstrip('/')}/{route}",
                "--out",
                str(target),
                "--width",
                "430",
                "--height",
                "932",
                "--dpr",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            problems.append(f"{name}: 截图失败（{message[-1] if message else '无输出'}）")
            if target.exists():
                target.unlink()
            continue
        rows.append(
            {
                "file": name,
                "url": f"{base_url.rstrip('/')}/{route}",
                **describe(target),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="生成第 5 周演示图")
    parser.add_argument("--out", required=True, help="输出目录，例如 output/week5")
    parser.add_argument("--base-url", default="", help="移动端地址；给出时同时截图")
    parser.add_argument("--content", default=str(CONTENT), help="atlas.json 路径")
    parser.add_argument("--samples", default=str(SAMPLES), help="公开预览目录")
    parser.add_argument("--sample-index", default=str(SAMPLE_INDEX), help="公开样例清单（含 SHA-256）")
    parser.add_argument("--native-dir", default="", help="原生裁片目录；同名文件优先于放大预览")
    parser.add_argument("--cdp", default=str(REPO / "scripts" / "cdp_screenshot.py"), help="截图脚本")
    parser.add_argument("--skip-shots", action="store_true", help="只生成演示输入与图表")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    content = json.loads(Path(args.content).read_text(encoding="utf-8"))
    cases = {case["landslideId"]: case for case in content["cases"]}
    allowed = public_ids()
    samples = sample_index(Path(args.sample_index))
    native_dir = Path(args.native_dir) if args.native_dir else None
    problems: list[str] = []

    manifest_path = out / "figures-manifest.json"
    previous: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = {item["file"]: item for item in old.get("demoInputs", [])}
        except (ValueError, KeyError):
            previous = {}

    manifest: dict = {"demoInputs": [], "figures": [], "problems": problems}
    sheet_inputs = []
    for sid, name, type_code in DEMO_CASES:
        case = cases[sid]
        number = int(sid.split("_")[1])
        if number not in allowed:
            problems.append(f"{sid}: 不在公开子集里，不能作为演示输入")
        replacement = replacement_of(name, out, native_dir, previous)
        if replacement is not None:
            path, mode = replacement
            if mode == "native":
                target = out / name
                if path.resolve() != target.resolve():
                    target.write_bytes(path.read_bytes())
                path = target
            meta = describe_native(path, case, mode)
        else:
            source = Path(args.samples) / f"{sid}.png"
            if not source.exists():
                problems.append(f"{sid}: 缺少公开预览 {source}")
                continue
            sample = samples.get(sid)
            if sample is None:
                problems.append(f"{sid}: 公开样例清单里没有这条，不能作为演示输入")
                continue
            source_problems = verify_preview(source, sample)
            if source_problems:
                problems.extend(source_problems)
                continue
            meta = demo_canvas(source, out / name, case)
        if meta["size"][0] > MAX_EDGE or max(meta["size"]) > MAX_EDGE or meta["bytes"] > MAX_BYTES:
            problems.append(f"{name}: 尺寸或体积超过上传上限")
        meta.update(
            {
                "sid": sid,
                "type": type_code,
                "subtitle": case["subtitle"],
                "path": str(out / name),
                "sourceScale": scale_of(case),
                "tile": case["tile"],
            }
        )
        manifest["demoInputs"].append({k: v for k, v in meta.items() if k != "path"})
        sheet_inputs.append(meta)

    demo_sheet(sheet_inputs, out / "fig-demo-input.png")
    chart_types(content, 2971, out / "chart-types.png")
    chart_scale(content, out / "chart-scale.png")
    manifest["figures"] += [
        describe(out / name) for name in ("fig-demo-input.png", "chart-types.png", "chart-scale.png")
    ]

    if args.base_url and not args.skip_shots:
        routes = {
            "shot-explore.png": "#/",
            "shot-atlas.png": "#/atlas",
            "shot-case-sl.png": "#/atlas/case/case-ls-0493",
            "shot-case-noimage.png": "#/atlas/case/case-ls-0591",
            "shot-compare.png": "#/atlas/compare",
            "shot-atlas-type.png": "#/atlas/type/SP",
        }
        manifest["screenshots"] = screenshots(
            args.base_url, out, Path(args.cdp), routes, problems
        )
        manifest["figures"] += [describe(out / name) for name in routes if (out / name).exists()]

    (out / "figures-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"演示图已生成：{out}（输入 {len(manifest['demoInputs'])} 张、图 {len(manifest['figures'])} 张，问题 {len(problems)} 条）"
    )
    for problem in problems:
        print(f"  - {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
