"""Task Scheduler bootstrap: verified author weights, then offline Mars-ViT Base CV.

Network is allowed only for the fixed Zenodo weight download. Four resumable
8 MiB segments are requested concurrently. The existing strict CPU preparation
and CUDA classifier runner are called in this same system-owned process only
after published size/MD5 and local SHA256 verification. No existing experiment
is restarted or overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime

ROOT = Path(__file__).resolve().parents[2]
if "scheduled-run" in sys.argv:
    sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages"))
WEIGHT_DIR = ROOT / "runs/research/mars_vit_weights_20261015_v1"
WEIGHT = WEIGHT_DIR / "mars-mae-dino-vit-base-v1.pth"
SEGMENTS = WEIGHT_DIR / "download_segments"
LEGACY = WEIGHT_DIR / "api_attempt1.partial"
OUT = ROOT / "runs/experiments/mars_vit_base_fourclass_20261015_v1"
RECEIPT = WEIGHT_DIR / "receipt.json"
OFFICIAL_FILE = "https://zenodo.org/records/18180801/files/mars-mae-dino-vit-base-v1.pth"
DOWNLOAD_URL = "https://zenodo.org/api/records/18180801/files/mars-mae-dino-vit-base-v1.pth/content"
SIZE = 341679467
MD5 = "a2b6c983546329a269f3e512dc2a762e"
MIB = 1024 * 1024
SEGMENT_SIZE = 8 * MIB
SEGMENT_COUNT = (SIZE + SEGMENT_SIZE - 1) // SEGMENT_SIZE
MAX_ATTEMPTS = 5


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(MIB), b""):
            result.update(block)
    return result.hexdigest()


def identity():
    # Do not import the model/offline helper before completing network work.
    import psutil
    parent = psutil.Process().parent()
    parent_exe = Path(parent.exe()).resolve() if parent else None
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    owned = bool(parent_exe and parent_exe.is_relative_to(system)
        and parent_exe.name.lower() in {"svchost.exe", "taskeng.exe", "taskhostw.exe"})
    return dict(pid=os.getpid(), parent_pid=os.getppid(),
        parent_executable=str(parent_exe), scheduler_owned=owned)


class Progress:
    def __init__(self, run_id, process):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.sizes = {}
        self.state = dict(status="starting", phase="checking_inputs", run_id=run_id,
            started_utc=utc(), process_independence=process, total_bytes=SIZE,
            segment_bytes=SEGMENT_SIZE, parallel_requests=4, source=DOWNLOAD_URL,
            variant="Base", planned_small_arm="unverified_not_replaced",
            model_started=False, gpu_used=False, temporary_system_awake=False,
            bootstrap_script_sha256=digest(__file__))

    def publish(self, phase=None, **fields):
        with self.lock:
            if phase is not None:
                self.state["phase"] = phase
            self.state.update(fields, updated_utc=utc(),
                elapsed_seconds=round(time.monotonic() - self.started, 3),
                retained_segment_bytes=sum(self.sizes.values()),
                retained_segment_mib=round(sum(self.sizes.values()) / MIB, 2))
            write_json(WEIGHT_DIR / "pipeline_status.json", self.state)

    def segment(self, index, count):
        with self.lock:
            self.sizes[index] = count
            self.state.update(updated_utc=utc(),
                elapsed_seconds=round(time.monotonic() - self.started, 3),
                retained_segment_bytes=sum(self.sizes.values()),
                retained_segment_mib=round(sum(self.sizes.values()) / MIB, 2))
            write_json(WEIGHT_DIR / "pipeline_status.json", self.state)

    def event(self, **value):
        with self.lock:
            with (WEIGHT_DIR / "pipeline_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(run_id=self.state["run_id"], utc=utc(), **value)) + "\n")
                handle.flush()
            print(json.dumps(value), flush=True)


def paths(index):
    start = index * SEGMENT_SIZE
    end = min(SIZE, start + SEGMENT_SIZE) - 1
    stem = SEGMENTS / f"segment_{index:04d}"
    files = stem.with_suffix(".partial"), stem.with_suffix(".bin"), stem.with_suffix(".sha256")
    exact = WEIGHT_DIR.resolve() / "download_segments"
    if SEGMENTS.resolve() != exact or any(path.resolve().parent != exact for path in files):
        raise ValueError("Segment path resolves outside the exact intended directory")
    return start, end, *files


def segment_count(index):
    start, end, partial, complete, sha = paths(index)
    if complete.exists() and partial.exists():
        raise ValueError(f"Conflicting preserved segment files: {index}")
    path = complete if complete.exists() else partial
    count = path.stat().st_size if path.exists() else 0
    if count > end - start + 1:
        raise ValueError(f"Oversized preserved segment: {index}")
    if complete.exists():
        if count != end - start + 1:
            raise ValueError(f"Incomplete segment marked complete: {index}")
        actual = digest(complete)
        if sha.exists() and actual != sha.read_text(encoding="ascii").strip():
            raise ValueError(f"Preserved segment digest changed: {index}")
        if not sha.exists():
            sha.write_text(actual + "\n", encoding="ascii")
    return count


def complete_segment(index):
    start, end, partial, complete, sha = paths(index)
    if partial.stat().st_size != end - start + 1:
        raise ValueError("Cannot mark incomplete segment complete")
    os.rename(partial, complete)  # Windows rename refuses an existing destination.
    sha.write_text(digest(complete) + "\n", encoding="ascii")


def check_foreground_retired():
    import psutil
    record = WEIGHT_DIR / "foreground_attempt.json"
    if record.exists():
        info = json.loads(record.read_text(encoding="utf-8"))
        pid = info.get("pid")
        if pid and psutil.pid_exists(pid):
            process = psutil.Process(pid)
            retired = info.get("retired_utc")
            if not retired or process.create_time() <= datetime.fromisoformat(retired).timestamp():
                raise RuntimeError(f"Original foreground downloader is still active: {pid}")


def seed_legacy(progress, legacy):
    if not legacy.exists():
        return
    legacy_size = legacy.stat().st_size
    if not 0 <= legacy_size <= SIZE:
        raise ValueError("Legacy partial has an impossible size")
    with legacy.open("rb") as source:
        for index in range(SEGMENT_COUNT):
            start, end, partial, complete, sha = paths(index)
            available = max(0, min(end - start + 1, legacy_size - start))
            count = segment_count(index)
            if available > count:
                source.seek(start + count)
                with partial.open("ab") as target:
                    remaining = available - count
                    while remaining:
                        block = source.read(min(MIB, remaining))
                        if not block:
                            raise IOError("Legacy partial changed while seeding")
                        target.write(block)
                        remaining -= len(block)
                        count += len(block)
                        target.flush()
                        os.fsync(target.fileno())
                        progress.segment(index, count)
                if count == end - start + 1:
                    complete_segment(index)
            progress.segment(index, count)
    if legacy.stat().st_size != legacy_size:
        raise IOError("Legacy downloader is changing the preserved partial")
    progress.event(event="legacy_seeded", path=str(legacy), bytes=legacy_size,
        legacy_preserved=True)


def retry_after(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        stamp = parsedate_to_datetime(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0.0, (stamp - datetime.now(timezone.utc)).total_seconds())


class RangeProtocolError(RuntimeError):
    pass


def fetch_segment(index, progress, deadline, cancel):
    start, end, partial, complete, sha = paths(index)
    if complete.exists():
        return
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if cancel.is_set():
            raise RuntimeError("Download canceled after another segment failed")
        count = partial.stat().st_size if partial.exists() else 0
        if count == end - start + 1:
            complete_segment(index)
            return
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise TimeoutError("Bounded download time expired; segments retained")
        offset = start + count
        request = Request(DOWNLOAD_URL, headers={"Range": f"bytes={offset}-{end}",
            "Accept-Encoding": "identity", "User-Agent": "Mars-KG-Vis-verified-weight-bootstrap/1"})
        delay = min(60.0, 2.0 ** attempt)
        try:
            with urlopen(request, timeout=min(45.0, remaining_time)) as response:
                final_url = urlsplit(response.geturl())
                if final_url.scheme != "https" or final_url.hostname != "zenodo.org":
                    raise RangeProtocolError("Download redirected outside official HTTPS Zenodo")
                expected_range = f"bytes {offset}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RangeProtocolError("Incorrect HTTP Range response: " + str(response.status)
                        + " " + str(response.headers.get("Content-Range")))
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != end - offset + 1:
                    raise RangeProtocolError("HTTP Range Content-Length differs")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise RangeProtocolError("Compressed range responses are prohibited")
                pending = end - offset + 1
                with partial.open("ab") as target:
                    since_publish = 0
                    while pending:
                        if cancel.is_set() or time.monotonic() >= deadline:
                            raise TimeoutError("Download canceled/time expired; partial retained")
                        block = response.read(min(256 * 1024, pending))
                        if not block:
                            raise IOError("Short HTTP Range body; partial retained")
                        target.write(block)
                        count += len(block)
                        pending -= len(block)
                        since_publish += len(block)
                        if since_publish >= MIB or not pending:
                            target.flush()
                            os.fsync(target.fileno())
                            progress.segment(index, count)
                            since_publish = 0
                    if response.read(1):
                        raise RangeProtocolError("HTTP Range body exceeds requested segment")
            complete_segment(index)
            progress.event(event="segment_complete", index=index, bytes=count, attempt=attempt)
            return
        except HTTPError as error:
            try:
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                instructed_delay = retry_after(error.headers.get("Retry-After"))
                if instructed_delay is not None:
                    delay = instructed_delay
                detail = f"HTTP {error.code}"
            finally:
                error.close()
        except RangeProtocolError:
            raise
        except (URLError, OSError) as error:
            detail = f"{type(error).__name__}: {error}"
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"Segment {index} failed after {MAX_ATTEMPTS} attempts: {detail}")
        if time.monotonic() + delay >= deadline:
            raise TimeoutError("Required retry backoff exceeds bounded download time")
        progress.event(event="segment_retry", index=index, attempt=attempt,
            retained_bytes=partial.stat().st_size if partial.exists() else 0,
            error=detail, retry_seconds=delay)
        if cancel.wait(delay):
            raise RuntimeError("Download canceled during retry backoff")


def download(progress, legacy, timeout):
    SEGMENTS.mkdir(exist_ok=True)
    existing = sum(segment_count(i) for i in range(SEGMENT_COUNT))
    assembly = WEIGHT_DIR / "assembled.download"
    assembly_bytes = assembly.stat().st_size if assembly.exists() else 0
    if assembly_bytes > SIZE:
        raise ValueError("Oversized preserved assembly")
    required = SIZE - existing + SIZE - assembly_bytes + 64 * MIB
    free = shutil.disk_usage(WEIGHT_DIR).free
    if free < required:
        raise OSError(f"Insufficient space for retained segments plus assembly: need {required}, free {free}")
    progress.publish("seeding_preserved_partial", status="running", disk_free_bytes=free,
        additional_disk_bytes_required=required)
    seed_legacy(progress, legacy)
    progress.publish("downloading_segments", network_allowed=True,
        download_timeout_seconds=timeout)
    deadline, cancel = time.monotonic() + timeout, threading.Event()
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="zenodo-range") as pool:
        futures = [pool.submit(fetch_segment, i, progress, deadline, cancel)
            for i in range(SEGMENT_COUNT)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            cancel.set()
            for future in futures:
                future.cancel()
            raise
    progress.publish("assembling_complete_segments", network_allowed=False)
    # A previous interrupted assembly is resumed only after its entire prefix is
    # compared with retained segments. Failed/corrupted assemblies are preserved.
    with assembly.open("a+b") as target:
        target.seek(0)
        old_bytes = assembly_bytes
        compared = 0
        for index in range(SEGMENT_COUNT):
            start, end, partial, complete, sha = paths(index)
            if segment_count(index) != end - start + 1 or not complete.exists():
                raise ValueError("Missing complete segment before assembly")
            with complete.open("rb") as source:
                for block in iter(lambda: source.read(MIB), b""):
                    overlap = min(len(block), max(0, old_bytes - compared))
                    if overlap and target.read(overlap) != block[:overlap]:
                        raise ValueError("Preserved assembly prefix differs; no corrupt fallback")
                    if overlap < len(block):
                        target.write(block[overlap:])
                    compared += len(block)
                    target.flush()
                    progress.publish(assembled_bytes=compared)
        target.flush()
        os.fsync(target.fileno())
    verify_bytes(assembly)
    if WEIGHT.exists():
        verify_bytes(WEIGHT)
        progress.event(event="existing_verified_weight_preserved", path=str(WEIGHT))
    else:
        os.rename(assembly, WEIGHT)


def verify_bytes(path):
    if path.stat().st_size != SIZE:
        raise ValueError("Published author file size does not match")
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(MIB), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != MD5:
        raise ValueError("Published author MD5 does not match; model will not launch")
    return sha.hexdigest()


def source_receipt(progress):
    progress.publish("verifying_published_bytes", network_allowed=False)
    sha = verify_bytes(WEIGHT)
    if RECEIPT.exists():
        source = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if not (Path(source["path"]).resolve() == WEIGHT.resolve()
            and source["md5"] == MD5 and source["sha256"] == sha
            and int(source["bytes"]) == SIZE and source["url"].split("?", 1)[0] == OFFICIAL_FILE):
            raise ValueError("Existing author source receipt differs; preserved without overwrite")
    else:
        metadata = ROOT / "runs/research/fourclass_literature_20261014_v1/mars_vit_source_metadata.json"
        author = metadata.with_name("mars_vit_author_README.md")
        source = dict(path=str(WEIGHT.resolve()), bytes=SIZE, md5=MD5, sha256=sha,
            url=OFFICIAL_FILE, download_url=DOWNLOAD_URL,
            record_url="https://zenodo.org/records/18180801", record_id=18180801,
            title="Mars-ViT Pretrained Models", variant="Base",
            license={"id": "cc-by-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
            verified_utc=utc(), source_metadata_path=str(metadata),
            source_metadata_sha256=digest(metadata), author_readme_path=str(author),
            author_readme_sha256=digest(author), published_md5_verified=True,
            published_size_verified=True, download_mode="4 parallel resumable 8 MiB HTTP Range segments",
            range_probe_before_launch="HTTP 206 bytes 0-1/341679467 verified by operator",
            planned_small_arm="unverified_not_replaced", bootstrap_script_sha256=digest(__file__))
        write_json(RECEIPT, source)
    progress.publish("author_weight_verified", verified_sha256=sha,
        verified_md5=MD5, verified_bytes=SIZE, receipt=str(RECEIPT))


def clean_segment_duplicates(progress):
    exact = SEGMENTS.resolve()
    if exact != WEIGHT_DIR.resolve() / "download_segments":
        raise ValueError("Resolved segments directory differs from exact intended directory")
    removed = []
    for index in range(SEGMENT_COUNT):
        for path in paths(index)[2:]:
            resolved = path.resolve()
            if resolved.parent != exact or not resolved.is_relative_to(exact):
                raise ValueError("Refuse cleanup outside own exact segments directory")
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    progress.event(event="verified_segment_duplicates_removed", files=removed,
        legacy_partial_preserved=True, recursive_delete=False)


def pipeline(args):
    import msvcrt
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = (WEIGHT_DIR / "pipeline.lock").open("a+b")
    if lock_handle.seek(0, os.SEEK_END) == 0:
        lock_handle.write(b"\0")
        lock_handle.flush()
    lock_handle.seek(0)
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock_handle.close()
        raise RuntimeError("Another independent pipeline holds the download lock")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    log_path = WEIGHT_DIR / f"bootstrap_{run_id}.log"
    awake = False
    progress = None
    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    process = identity()
                    progress = Progress(run_id, process)
                    progress.publish(bootstrap_log=str(log_path))
                    if not process["scheduler_owned"]:
                        raise RuntimeError("Launch the base interpreter directly from Windows Task Scheduler")
                    out = args.out.resolve()
                    if not out.is_relative_to((ROOT / "runs/experiments").resolve()):
                        raise ValueError("Experiment must remain inside project runs/experiments")
                    if out.exists():
                        raise FileExistsError("Prior experiment preserved; use a fresh --out, no automatic restart")
                    check_foreground_retired()
                    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                        raise ctypes.WinError()
                    awake = True
                    progress.publish(temporary_system_awake=True)
                    if not WEIGHT.exists():
                        download(progress, args.legacy_partial.resolve(), args.download_timeout_seconds)
                    source_receipt(progress)
                    clean_segment_duplicates(progress)
                    # All download threads have joined. From this point the existing
                    # runner forbids networking and strictly checks author tensors.
                    progress.publish("strict_cpu_preparation", status="running_offline",
                        network_allowed=False, experiment=str(out))
                    import run_mars_vit_classification as M
                    if (M.OFFICIAL_SIZE, M.OFFICIAL_MD5, M.OFFICIAL_FILE) != (SIZE, MD5, OFFICIAL_FILE):
                        raise ValueError("Bootstrap and strict runner author provenance differs")
                    M.prepare(out, WEIGHT, RECEIPT)
                    progress.publish("offline_cuda_features_and_grouped_classifiers", model_started=True,
                        gpu_requested=True, gpu_used=None,
                        runtime_progress=str(out / "features/status.json"))
                    with (out / "training.log").open("x", encoding="utf-8", buffering=1) as stdout:
                        with (out / "training.err.log").open("x", encoding="utf-8", buffering=1) as stderr:
                            with redirect_stdout(stdout), redirect_stderr(stderr):
                                M.run(out)
                    progress.publish("complete", status="completed_internal_catalog_comparison",
                        gpu_used=True, final_classifier_exported=False,
                        external_val_test_read=False, finished_utc=utc())
                except BaseException as error:
                    traceback.print_exc()
                    if progress:
                        progress.publish(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
                            else "failed", error_type=type(error).__name__, error=str(error),
                            finished_utc=utc())
                    raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if progress:
            progress.publish(temporary_system_awake=False)
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        lock_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["scheduled-run"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--legacy-partial", type=Path, default=LEGACY)
    parser.add_argument("--download-timeout-seconds", type=int, default=10800)
    args = parser.parse_args()
    if not 1 <= args.download_timeout_seconds <= 10800:
        parser.error("Download timeout must be between 1 and 10800 seconds (maximum 3 hours)")
    pipeline(args)


if __name__ == "__main__":
    main()
