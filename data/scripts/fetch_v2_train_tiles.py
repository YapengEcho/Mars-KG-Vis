# -*- coding: utf-8 -*-
"""Restore catalogued v2 train CTX tiles, verify, then remove their ZIPs.

Downloads are resumable in a bounded number of byte-range connections.
Existing TIFFs are checked for geometry and truncated data before skipping.
No val/test/final-eval tile is downloaded or split assignment changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import threading
import time
import zipfile

import tifffile

BASE = Path(__file__).resolve().parents[2]
CTX = BASE / "data/raw/ctx_tiles"
MANIFEST = BASE / "data/raw/ctx_tiles_manifest.json"
PLAN = BASE / "data/raw/tile_split_plan_flval_v2.json"
FINAL = BASE / "data/raw/final_eval_tiles_frozen.json"
NO_CATALOG = {"E-072_N12", "E-076_N12"}


def train_tiles(plan_path=PLAN, manifest_path=MANIFEST, final_path=FINAL):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    catalog = {r["tile"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))}
    tiles = sorted(t for t, s in plan.items() if s == "train")
    missing = set(tiles) - set(catalog)
    if len(tiles) != 33 or missing != NO_CATALOG:
        raise ValueError(f"v2/catalog changed: train={len(tiles)}, absent={sorted(missing)}")
    frozen = set(json.loads(final_path.read_text(encoding="utf-8"))["tiles"])
    if frozen & set(tiles):
        raise ValueError("train plan overlaps frozen final-eval tiles")
    for name in set(tiles) & set(catalog):
        row = catalog[name]
        expected = f"MurrayLab_GlobalCTXMosaic_V01_{name}.zip"
        if row["zip"] != expected or row["url"] != f"https://murray-lab.caltech.edu/CTX/V01/tiles/{expected}":
            raise ValueError(f"unexpected catalog path/URL: {name}")
    return sorted(set(tiles) & set(catalog))


def verify_tif(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if len(page.shape) != 2 or min(page.shape) < 1024:
            raise ValueError(f"unexpected CTX shape: {page.shape}")
        if page.dtype.name != "uint8" or not page.tags.get(33550) or not page.tags.get(33922):
            raise ValueError("CTX dtype/georeference invalid")
        offsets, counts = page.dataoffsets, page.databytecounts
        if not offsets or any(a + b > path.stat().st_size for a, b in zip(offsets, counts)):
            raise ValueError("TIFF pixel data truncated")
        return {"path": str(path), "bytes": path.stat().st_size, "shape": list(page.shape),
                "pixel_scale": list(page.tags[33550].value), "tiepoint": list(page.tags[33922].value)}


def existing_tif(root: Path, tile: str):
    paths = list(root.glob(f"*/MurrayLab_CTX_V01_{tile}_Mosaic.tif"))
    if len(paths) > 1:
        raise ValueError(f"multiple copies of tile {tile}")
    if not paths:
        return None
    try:
        return verify_tif(paths[0])
    except (ValueError, OSError, tifffile.TiffFileError):
        return None


def confined(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"path outside intended CTX directory: {path}")
    return path


def segments(total: int, count: int):
    if total <= 0 or count <= 0:
        raise ValueError("positive file size and connection count required")
    count = min(count, total)
    return [(total * i // count, total * (i + 1) // count - 1) for i in range(count)]


def download_segment(url, path, start, end, total, retries=5):
    expected = end - start + 1
    for attempt in range(retries):
        present = path.stat().st_size if path.exists() else 0
        if present == expected:
            return
        if present > expected:
            raise ValueError(f"oversized segment: {path}")
        begin = start + present
        # Windows curl uses the OS certificate store. Keep TLS validation,
        # and validate exact range headers before appending received bytes.
        chunk = path.with_suffix(".receiving")
        headers = path.with_suffix(".headers")
        command = [
            "curl.exe", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "20", "--max-time", "180",
            "--speed-time", "60", "--speed-limit", "1024",
            "--range", f"{begin}-{end}", "--header", "Accept-Encoding: identity",
            "--dump-header", str(headers), "--output", str(chunk), url,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=210)
        except subprocess.TimeoutExpired:
            # subprocess kills the stuck child; validated partial bytes below
            # still become a resumable prefix, while missing headers never do.
            result = subprocess.CompletedProcess(command, 124, stderr="segment process timed out")
        header_text = headers.read_text(encoding="latin-1") if headers.exists() else ""
        statuses = re.findall(r"^HTTP/\S+\s+(\d+)", header_text, re.MULTILINE)
        ranges = re.findall(r"^Content-Range:\s*bytes (\d+)-(\d+)/(\d+)",
                            header_text, re.IGNORECASE | re.MULTILINE)
        valid = (statuses and statuses[-1] == "206" and ranges
                 and tuple(map(int, ranges[-1])) == (begin, end, total))
        if valid and chunk.exists():
            if chunk.stat().st_size > end - begin + 1:
                raise ValueError("server sent excess range data")
            with chunk.open("rb") as src, path.open("ab") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            chunk.unlink()
            if path.stat().st_size == expected:
                if headers.exists():
                    headers.unlink()
                return
        elif result.returncode == 0:
            raise ValueError("server did not honor the exact byte range")
        if attempt == retries - 1 and result.returncode:
            raise IOError(f"curl {result.returncode}: {result.stderr.strip()}")
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 15))
    raise IOError(f"incomplete range: {path}")


def download_zip(row, root, connections):
    zip_path = confined(root, root / row["zip"])
    total = row["size_bytes"]
    if zip_path.exists() and zip_path.stat().st_size == total:
        return zip_path
    if zip_path.exists() and zip_path.stat().st_size > total:
        raise ValueError(f"archive larger than catalog: {zip_path}")
    parts_dir = confined(root, root / "_v2_download_parts" / row["tile"])
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = segments(total, connections)
    # Seed segmented resume from the previous single-connection ZIP prefix.
    if zip_path.exists():
        with zip_path.open("rb") as src:
            for index, (start, end) in enumerate(ranges):
                part = parts_dir / f"part_{index:02d}_{start}_{end}.bin"
                present = part.stat().st_size if part.exists() else 0
                available = min(end - start + 1, max(0, zip_path.stat().st_size - start))
                if present >= available:
                    continue
                src.seek(start + present)
                with part.open("ab") as dst:
                    remaining = available - present
                    while remaining:
                        block = src.read(min(8 << 20, remaining))
                        if not block:
                            raise IOError("existing prefix changed during resume")
                        dst.write(block)
                        remaining -= len(block)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(download_segment, row["url"],
                   parts_dir / f"part_{i:02d}_{a}_{b}.bin", a, b, total)
                   for i, (a, b) in enumerate(ranges)]
        for future in as_completed(futures):
            future.result()
    assembled = confined(root, root / (row["zip"] + ".assembling"))
    with assembled.open("wb") as dst:
        for i, (a, b) in enumerate(ranges):
            with (parts_dir / f"part_{i:02d}_{a}_{b}.bin").open("rb") as src:
                shutil.copyfileobj(src, dst, 8 << 20)
    if assembled.stat().st_size != total:
        raise ValueError("assembled archive size mismatch")
    os.replace(assembled, zip_path)
    for i, (a, b) in enumerate(ranges):
        confined(root, parts_dir / f"part_{i:02d}_{a}_{b}.bin").unlink()
    parts_dir.rmdir()
    return zip_path


def extract_verified(row, zip_path, root):
    """Streaming extraction checks ZIP CRC; commit only complete member files."""
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        expected_name = f"MurrayLab_CTX_V01_{row['tile']}_Mosaic.tif"
        mosaics = [info for info in infos if PurePosixPath(info.filename).name == expected_name]
        if len(mosaics) != 1:
            raise ValueError("archive must contain exactly the expected Mosaic.tif")
        targets = []
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ValueError("unexpected archive member path")
            target = confined(root, root.joinpath(*name.parts))
            targets.append((info, target))
        free = shutil.disk_usage(root).free
        if free < sum(i.file_size for i in infos) + (2 << 30):
            raise OSError("insufficient disk space for extraction plus 2 GiB reserve")
        for info, target in targets:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = confined(root, target.with_name(target.name + ".extracting"))
            with archive.open(info) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            if temporary.stat().st_size != info.file_size:
                raise ValueError("extracted file size mismatch")
            if target.name == expected_name:
                verify_tif(temporary)
            os.replace(temporary, target)
    verified = existing_tif(root, row["tile"])
    if verified is None:
        raise ValueError("extraction produced no valid TIFF")
    confined(root, zip_path).unlink()
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--connections", type=int, default=4)
    parser.add_argument("--report", type=Path, default=BASE / "runs/v2_train_restore_status.json")
    parser.add_argument("--tiles", nargs="*", help="Optional subset of catalogued v2 train tiles")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or not 1 <= args.connections <= 8:
        parser.error("workers must be 1..3, connections 1..8")
    wanted = train_tiles()
    if args.tiles:
        if set(args.tiles) - set(wanted):
            parser.error("subset includes a non-catalogued/non-train tile")
        wanted = [t for t in wanted if t in args.tiles]
    by_tile = {r["tile"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    CTX.mkdir(parents=True, exist_ok=True)
    status = {"status": "checking", "workers": args.workers, "connections_per_tile": args.connections,
              "wanted": wanted, "no_catalog": sorted(NO_CATALOG), "tiles": {}, "failed": {}}
    state_lock, extract_lock = threading.Lock(), threading.Lock()

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        status["updated_epoch"] = time.time()
        temp = args.report.with_suffix(".tmp")
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, args.report)

    def process(name):
        row = by_tile[name]
        verified = existing_tif(CTX, name)
        if verified is None:
            with state_lock:
                status["tiles"][name] = {"status": "downloading"}
                save()
            print(f"DOWNLOAD {name}", flush=True)
            zip_path = download_zip(row, CTX, args.connections)
            with state_lock:
                status["tiles"][name] = {"status": "waiting_extract"}
                save()
            with extract_lock:
                print(f"EXTRACT {name}", flush=True)
                verified = extract_verified(row, zip_path, CTX)
        else:
            zip_path = confined(CTX, CTX / row["zip"])
            if zip_path.exists():
                zip_path.unlink()
        with state_lock:
            status["tiles"][name] = {"status": "verified", **verified}
            save()
        print(f"VERIFIED {name}", flush=True)

    if args.status_only:
        for name in wanted:
            verified = existing_tif(CTX, name)
            status["tiles"][name] = {"status": "verified", **verified} if verified else {"status": "missing"}
        status["status"] = "inventory_only"
        save()
        print(json.dumps({"verified": sum(r["status"] == "verified" for r in status["tiles"].values()),
                          "wanted": len(wanted), "free_gib": shutil.disk_usage(CTX).free / 2**30}, indent=2))
        return
    # One writer across Python interpreters, released even after exceptions.
    import msvcrt
    lock_path = CTX / "_v2_restore.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            status["status"] = "running"
            save()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process, name): name for name in wanted}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        with state_lock:
                            status["failed"][name] = str(exc)
                            status["tiles"][name] = {"status": "failed", "error": str(exc)}
                            save()
                        print(f"FAILED {name}: {exc}", flush=True)
            status["status"] = "failed" if status["failed"] else "complete"
            save()
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    print(f"FINISHED {status['status']}: {len(wanted) - len(status['failed'])}/{len(wanted)}", flush=True)
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
