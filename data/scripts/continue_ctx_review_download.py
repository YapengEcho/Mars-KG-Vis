"""Download an explicitly audited CTX queue and build an independent review pack."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import shapefile

from build_v2_train_center_review import build_complete_pack, finalize, save_json
from check_tile_connectivity import tiles_of_bbox
from fetch_v2_train_tiles import BASE, CTX, MANIFEST, FINAL, existing_tif, download_zip, extract_verified
from make_centered_review_pack import SHP, map_source_records


def protected_tiles():
    protected = set(json.loads(FINAL.read_text(encoding="utf-8"))["tiles"])
    for name in ("tile_split_plan_flval.json", "tile_split_plan_flval_v2.json",
                 "tile_split_plan_flval_train_expansion.json"):
        plan = json.loads((BASE / "data/raw" / name).read_text(encoding="utf-8"))
        protected.update(t for t, split in plan.items() if split in {"val", "test"})
    return protected


def validate_queue(wanted, catalog, touches, protected):
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("download queue must be nonempty and unique")
    for tile in wanted:
        row = catalog[tile]
        archive = f"MurrayLab_GlobalCTXMosaic_V01_{tile}.zip"
        if (row["zip"] != archive or row["url"] !=
                f"https://murray-lab.caltech.edu/CTX/V01/tiles/{archive}"):
            raise ValueError(f"unexpected catalog URL: {tile}")
        ids = [sid for sid, hits in touches.items() if tile in hits]
        if tile in protected or any(touches[sid] & protected for sid in ids):
            raise ValueError(f"protected tile or target overlap: {tile}")
        if not ids:
            raise ValueError(f"no catalogued landslide target: {tile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--connections", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or not 1 <= args.connections <= 8:
        parser.error("workers must be 1..3, connections must be 1..8")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    wanted = plan["tiles"]
    catalog = {r["tile"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    with shapefile.Reader(str(SHP)) as reader:
        sources = map_source_records(reader.shapeRecords())
    touches = {sid: tiles_of_bbox(*r.shape.bbox) for sid, r in sources.items()}
    validate_queue(wanted, catalog, touches, protected_tiles())
    # New files remain independent candidates; never rewrite frozen split plans.
    signatures = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (args.plan, MANIFEST, SHP, SHP.with_suffix(".dbf"),
                            SHP.with_suffix(".shx"), FINAL)}
    state = {"status": "running", "wanted": wanted, "workers": args.workers,
             "connections_per_tile": args.connections, "tiles": {}, "failed": {},
             "out": str(args.out.resolve()), "input_sha256": signatures,
             "purpose": "new review candidates; no training or split changes"}
    state_lock, extract_lock, render_lock = threading.Lock(), threading.Lock(), threading.Lock()

    def save():
        state["updated_epoch"] = time.time()
        save_json(args.report, state)

    def mark(tile, status, **details):
        with state_lock:
            state["tiles"][tile] = {"status": status, **details}
            save()

    def process(tile):
        verified = existing_tif(CTX, tile)
        if verified is None:
            mark(tile, "downloading")
            print(f"DOWNLOAD {tile}", flush=True)
            archive = download_zip(catalog[tile], CTX, args.connections)
            mark(tile, "waiting_extract")
            with extract_lock:
                verified = extract_verified(catalog[tile], archive, CTX)
        mark(tile, "verified", **verified)
        print(f"VERIFIED {tile}", flush=True)
        out = args.out / "tiles" / tile
        with render_lock:
            marker = out / "centered_review_manifest.json"
            if not marker.exists():
                mark(tile, "building_review", **verified)
                out.mkdir(parents=True, exist_ok=True)
                result = build_complete_pack(tile, out)
            else:
                result = {"count": json.loads(marker.read_text(encoding="utf-8"))["count"]}
        mark(tile, "review_complete", **verified, review_count=result["count"])
        print(f"REVIEW_COMPLETE {tile} {result['count']}", flush=True)

    # Share the existing downloader lock to avoid concurrent writes to CTX archives.
    import msvcrt
    CTX.mkdir(parents=True, exist_ok=True)
    with (CTX / "_v2_restore.lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            save()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                tasks = {pool.submit(process, tile): tile for tile in wanted}
                for task in as_completed(tasks):
                    tile = tasks[task]
                    try:
                        task.result()
                    except Exception as exc:
                        with state_lock:
                            state["failed"][tile] = str(exc)
                            state["tiles"][tile] = {"status": "failed", "error": str(exc)}
                            save()
                        print(f"FAILED {tile}: {exc}", flush=True)
            if not state["failed"]:
                finalize(args.out, wanted, signatures)
                summary_path = args.out / "centered_review_manifest.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["source_queue"] = str(args.plan.resolve())
                summary["split_policy"] = "candidate train sources; original frozen plans unchanged"
                save_json(summary_path, summary)
                save_json(args.out / "progress.json", {"status": "complete", "completed": wanted})
            state["status"] = "failed" if state["failed"] else "complete"
            save()
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    print(f"FINISHED {state['status']}", flush=True)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
