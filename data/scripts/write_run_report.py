import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

CHUNK = 1 << 20
REPO = Path(__file__).resolve().parents[2]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def git_info():
    def run(args):
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO)] + args,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    head = run(["rev-parse", "HEAD"])
    branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run(["status", "--porcelain"])
    return {
        "head": head,
        "branch": branch,
        "dirty": bool(dirty),
        "dirty_files": len(dirty.splitlines()) if dirty else 0,
    }


def dep_versions():
    out = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "numpy"):
        try:
            out[name] = pkg_version(name)
        except Exception:
            out[name] = None
    return out


def load_yaml(path):
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve_manifest(value, manifest_dir):
    p = Path(value)
    if p.is_file():
        return p
    return manifest_dir / f"{value}.json"


def self_hash(report):
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_report(name, manifest=None, manifest_dir="data/manifest", args_path=None, seed=None, weights=None, note=None):
    report = {
        "experiment": name,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_info(),
        "deps": dep_versions(),
    }

    seed_value = seed
    if args_path:
        args_file = Path(args_path)
        if args_file.is_file():
            payload = load_yaml(args_file)
            if seed_value is None and "seed" in payload:
                seed_value = payload["seed"]
            report["args"] = {"path": args_file.as_posix(), "sha256": sha256_file(args_file)}
        else:
            report["args"] = {"path": args_file.as_posix(), "sha256": None, "missing": True}
    report["seed"] = seed_value

    if manifest:
        manifest_path = resolve_manifest(manifest, Path(manifest_dir))
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            report["manifest"] = {
                "path": manifest_path.as_posix(),
                "dataset": data.get("dataset"),
                "manifest_sha256": data.get("manifest_sha256"),
                "rollup_sha256": data.get("rollup_sha256"),
            }
        else:
            report["manifest"] = {"path": manifest_path.as_posix(), "missing": True}

    entries = []
    for item in weights or []:
        weight_path = Path(item)
        if weight_path.is_file():
            entries.append(
                {
                    "path": weight_path.as_posix(),
                    "sha256": sha256_file(weight_path),
                    "bytes": weight_path.stat().st_size,
                }
            )
        else:
            entries.append({"path": weight_path.as_posix(), "sha256": None, "missing": True})
    report["weights"] = entries

    if note:
        report["note"] = note

    report["report_sha256"] = self_hash(report)
    return report


def write_report(report, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def print_report(report, out_path):
    print(f"[written] {Path(out_path).as_posix()}")
    print(f"    git            = {report['git'].get('head')}  dirty={report['git'].get('dirty')}")
    print(
        f"    deps           = ultralytics={report['deps'].get('ultralytics')} "
        f"torch={report['deps'].get('torch')} python={report['deps'].get('python')}"
    )
    if "manifest" in report:
        print(
            f"    manifest       = {report['manifest'].get('dataset')}  "
            f"rollup={report['manifest'].get('rollup_sha256')}"
        )
    if "args" in report:
        print(f"    args           = {report['args'].get('path')}  sha256={report['args'].get('sha256')}")
    print(f"    seed           = {report['seed']}")
    print(f"    weights        = {len(report['weights'])} file(s)")
    print(f"    report_sha256  = {report['report_sha256']}")


def main():
    ap = argparse.ArgumentParser(description="Write a run fingerprint report for one training/eval run.")
    ap.add_argument("--name", required=True, help="experiment name")
    ap.add_argument("--manifest", help="dataset name or manifest json path")
    ap.add_argument("--manifest-dir", default="data/manifest", help="directory holding manifests")
    ap.add_argument("--args", help="path to args.yaml of the run")
    ap.add_argument("--seed", type=int, help="random seed; read from --args when omitted")
    ap.add_argument("--weights", nargs="*", default=[], help="weight file(s) to hash")
    ap.add_argument("--note", help="free-form note")
    ap.add_argument("--out", help="output json path; defaults to runs/<name>_run_report.json")
    args = ap.parse_args()

    report = build_report(
        name=args.name,
        manifest=args.manifest,
        manifest_dir=args.manifest_dir,
        args_path=args.args,
        seed=args.seed,
        weights=args.weights,
        note=args.note,
    )
    out_path = Path(args.out) if args.out else REPO / "runs" / f"{args.name}_run_report.json"
    write_report(report, out_path)
    print_report(report, out_path)


if __name__ == "__main__":
    main()
