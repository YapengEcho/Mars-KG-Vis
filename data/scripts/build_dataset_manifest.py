import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "val", "test")
CHUNK = 1 << 20
TILE_RE = re.compile(r"^(E-?\d+_N-?\d+)")
GENERATOR = Path(__file__).resolve()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def count_rows(path):
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def split_of(path):
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] in ("images", "labels"):
        return parts[1]
    return None


def collect_files(dataset_dir):
    records = []
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(dataset_dir).as_posix()
            records.append(
                {
                    "path": rel,
                    "split": split_of(rel),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )
    return records


def rollup(records):
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r['path']}|{r['bytes']}|{r['sha256']}\n".encode("utf-8"))
    return h.hexdigest()


def summarize_splits(dataset_dir):
    splits = {}
    for split in SPLITS:
        img_dir = dataset_dir / "images" / split
        lab_dir = dataset_dir / "labels" / split
        images = [p for p in sorted(img_dir.glob("*")) if p.is_file()] if img_dir.exists() else []
        labels = sorted(lab_dir.glob("*.txt")) if lab_dir.exists() else []
        if not images and not labels:
            continue
        rows = 0
        empty = 0
        for lab in labels:
            c = count_rows(lab)
            rows += c
            if c == 0:
                empty += 1
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "label_rows": rows,
            "empty_labels": empty,
        }
    return splits


def extract_tiles(records):
    tiles = set()
    for r in records:
        m = TILE_RE.match(Path(r["path"]).name)
        if m:
            tiles.add(m.group(1))
    return sorted(tiles)


def self_hash(manifest):
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize_duplicates(records):
    images = [r for r in records if r["path"].startswith("images/")]
    by_hash = {}
    for r in images:
        by_hash.setdefault(r["sha256"], []).append(r)
    groups = {h: v for h, v in by_hash.items() if len(v) > 1}
    cross = 0
    redundant = 0
    samples = []
    for h, v in groups.items():
        if len({r["split"] for r in v}) > 1:
            cross += 1
        redundant += len(v) - 1
        if len(samples) < 10:
            samples.append(sorted(r["path"] for r in v))
    return {
        "images": len(images),
        "unique_hashes": len(by_hash),
        "duplicate_groups": len(groups),
        "redundant_images": redundant,
        "cross_split_groups": cross,
        "samples": samples,
    }


def build(dataset_dir, out_dir):
    records = collect_files(dataset_dir)
    manifest = {
        "dataset": dataset_dir.name,
        "root": f"data/processed/{dataset_dir.name}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "script": "data/scripts/build_dataset_manifest.py",
            "sha256": sha256_file(GENERATOR),
        },
        "tiles": extract_tiles(records),
        "duplicates": summarize_duplicates(records),
        "summary": {
            "files": len(records),
            "bytes_total": sum(r["bytes"] for r in records),
            "splits": summarize_splits(dataset_dir),
        },
        "rollup_sha256": rollup(records),
        "files": records,
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_dir.name}.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, manifest


def verify(manifest_path, dataset_dir):
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_map = {r["path"]: (r["bytes"], r["sha256"]) for r in old.get("files", [])}
    new = collect_files(dataset_dir)
    new_map = {r["path"]: (r["bytes"], r["sha256"]) for r in new}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    changed = sorted(k for k in set(old_map) & set(new_map) if old_map[k] != new_map[k])
    new_rollup = rollup(new)
    return {
        "manifest": str(manifest_path),
        "dataset": dataset_dir.name,
        "rollup_match": new_rollup == old.get("rollup_sha256"),
        "self_hash_match": self_hash(old) == old.get("manifest_sha256"),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def prefix_of(path):
    parts = path.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def compare(manifest_a, manifest_b):
    a = json.loads(Path(manifest_a).read_text(encoding="utf-8"))
    b = json.loads(Path(manifest_b).read_text(encoding="utf-8"))
    am = {r["path"]: r["sha256"] for r in a["files"]}
    bm = {r["path"]: r["sha256"] for r in b["files"]}
    shared = set(am) & set(bm)
    changed = sorted(k for k in shared if am[k] != bm[k])
    same = sorted(k for k in shared if am[k] == bm[k])
    added = sorted(set(bm) - set(am))
    removed = sorted(set(am) - set(bm))

    def group(paths):
        g = {}
        for p in paths:
            g[prefix_of(p)] = g.get(prefix_of(p), 0) + 1
        return dict(sorted(g.items()))

    return {
        "a": a["dataset"],
        "b": b["dataset"],
        "a_rollup": a["rollup_sha256"],
        "b_rollup": b["rollup_sha256"],
        "identical": len(same),
        "changed_by_prefix": group(changed),
        "added_by_prefix": group(added),
        "removed_by_prefix": group(removed),
        "changed_samples": changed[:20],
    }


def resolve_dataset(name_or_path, processed_root):
    p = Path(name_or_path)
    if p.is_dir():
        return p
    return processed_root / name_or_path


def main():
    ap = argparse.ArgumentParser(description="Build, verify, or compare SHA256 manifests for dataset directories.")
    ap.add_argument("--dataset", nargs="+", help="dataset name(s) under --processed-root, or explicit paths")
    ap.add_argument("--compare", nargs=2, metavar=("MANIFEST_A", "MANIFEST_B"), help="compare two manifests by per-file sha256")
    ap.add_argument("--processed-root", default="data/processed", help="root that dataset names resolve against")
    ap.add_argument("--out-dir", default="data/manifest", help="output directory for manifest json")
    ap.add_argument("--verify", action="store_true", help="re-hash and diff against existing manifest instead of writing")
    ap.add_argument("--duplicates", action="store_true", help="report content-duplicate groups from an existing manifest")
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)

    if args.compare:
        res = compare(Path(args.compare[0]), Path(args.compare[1]))
        print(f"A = {res['a']}")
        print(f"    rollup={res['a_rollup']}")
        print(f"B = {res['b']}")
        print(f"    rollup={res['b_rollup']}")
        print(f"identical files = {res['identical']}")
        print(f"changed_by_prefix = {res['changed_by_prefix']}")
        print(f"added_by_prefix   = {res['added_by_prefix']}")
        print(f"removed_by_prefix = {res['removed_by_prefix']}")
        for name in res["changed_samples"]:
            print(f"    changed: {name}")
        return

    if not args.dataset:
        ap.error("--dataset is required unless --compare is used")

    for item in args.dataset:
        dataset_dir = resolve_dataset(item, processed_root)
        if not dataset_dir.is_dir():
            print(f"[skip] not a directory: {dataset_dir}")
            continue

        if args.duplicates:
            manifest_path = out_dir / f"{dataset_dir.name}.json"
            if not manifest_path.exists():
                print(f"[skip] no manifest: {manifest_path}")
                continue
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            d = data.get("duplicates")
            if d is None:
                print(f"[skip] no duplicates section: {manifest_path}")
                continue
            print(
                f"[{data['dataset']}] images={d['images']} unique={d['unique_hashes']} "
                f"groups={d['duplicate_groups']} redundant={d['redundant_images']} cross_split={d['cross_split_groups']}"
            )
            for group in d["samples"]:
                print("    " + " | ".join(group))
            continue

        if args.verify:
            manifest_path = out_dir / f"{dataset_dir.name}.json"
            if not manifest_path.exists():
                print(f"[skip] no manifest: {manifest_path}")
                continue
            res = verify(manifest_path, dataset_dir)
            status = "MATCH" if res["rollup_match"] else "DRIFT"
            print(f"[{status}] {res['dataset']}  manifest_self_hash={'ok' if res['self_hash_match'] else 'bad'}  added={len(res['added'])} removed={len(res['removed'])} changed={len(res['changed'])}")
            for key in ("added", "removed", "changed"):
                for name in res[key][:20]:
                    print(f"    {key}: {name}")
            continue

        out_path, manifest = build(dataset_dir, out_dir)
        s = manifest["summary"]
        print(f"[built] {out_path}")
        print(f"    files={s['files']}  bytes={s['bytes_total']}  tiles={len(manifest['tiles'])}")
        for split, info in s["splits"].items():
            print(f"    {split}: images={info['images']} labels={info['labels']} rows={info['label_rows']} empty={info['empty_labels']}")
        d = manifest["duplicates"]
        print(f"    dup: images={d['images']} unique={d['unique_hashes']} groups={d['duplicate_groups']} redundant={d['redundant_images']} cross_split={d['cross_split_groups']}")
        print(f"    rollup_sha256={manifest['rollup_sha256']}")
        print(f"    manifest_sha256={manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
