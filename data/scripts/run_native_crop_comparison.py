"""Independent offline paired comparison of old and native CTX classifier crops.

Freeze crop geometry and count-only folds before native pixels or model scores.
Build native inputs, reuse verified old frozen vectors, and refit both six-head
screens on exactly the same targets and folds. No final model is exported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if "scheduled-run" in sys.argv:
    sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages"))
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true")

import numpy as np
import extract_manual_classification_features as C
import run_full_pretrained_detector as F
import build_native_classification_dataset as N
import prepare_native_crop_comparison_protocol as Q

OUT = ROOT / "runs/experiments/native_crop_comparison_20261015_v1"
MASTER = ROOT / "runs/research/native_crop_comparison_protocol_20261015_v1/master_plan.json"
NATIVE_PLAN = ROOT / "runs/research/native_crop_comparison_preparation_20261015_v1/native_build_plan.json"
NATIVE_MANIFEST = ROOT / "data/processed/manual_stage2_native_20261015_v1/manifest.json"
NATIVE_STATUS = NATIVE_MANIFEST.parent / "status.json"
OLD = ROOT / "runs/experiments/fourclass_corrected_groups_20261015_v2"
PRIMARY = dict(encoder="dinov2", head="logistic_regression", balance_mode="balanced", C=1.0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    C.write_json_atomic(path, value)


def matrix_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(str(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "transformers", "scikit-learn", "pillow", "numpy", "tifffile")}


def prepare(out):
    master = Q.verify_master(MASTER)
    original, X, y, _, receipt = C.load_verified_features(OLD / "plan.json", OLD / "dinov2/features")
    if X.shape != (1083, 768) or not np.array_equal(y, [r["class_id"] for r in master["rows"]]):
        raise ValueError("Require the existing full1083 frozen DINOv2 vectors and same ordered labels")
    for row, before in zip(master["rows"], original["rows"]):
        if row["source_id"] != before["source_id"] or row["catalog_class"] != before["catalog_class"]:
            raise ValueError("Comparison changed target IDs or catalog types")
    checkpoint = Path(receipt["checkpoint"])
    weights = {name: C.sha256(checkpoint / name) for name in ("config.json", "model.safetensors")}
    if weights != receipt["checkpoint_files_sha256"]:
        raise ValueError("Frozen DINOv2 checkpoint differs from the baseline")
    sources = [Path(__file__), Path(C.__file__), Path(F.__file__), Path(N.__file__), Path(Q.__file__)]
    sources += [ROOT / "data/scripts" / name for name in (
        "build_native_classification_crop_pilot.py", "recover_reviewed_mosaic_samples.py",
        "cross_tile_neighbor_review.py", "export_manual_detection_dataset.py",
        "make_centered_review_pack.py", "train_stage2_cls.py",
        "prepare_manual_classification_protocol.py", "start_native_crop_comparison.ps1")]
    code = {str(p.resolve()): C.sha256(p) for p in sources}
    out = C.new_directory(out)
    spec = dict(status="locked_before_native_pixel_build_and_paired_scores", created_utc=utc(),
                scope=C.POLICY, master_plan=str(MASTER), master_plan_sha256=C.sha256(MASTER),
                native_build_plan=str(NATIVE_PLAN), native_build_plan_sha256=C.sha256(NATIVE_PLAN),
                native_output_manifest=str(NATIVE_MANIFEST), native_status=str(NATIVE_STATUS),
                source_code_sha256=code, old_plan=str(OLD / "plan.json"), old_plan_sha256=C.sha256(OLD / "plan.json"),
                old_feature_dir=str(OLD / "dinov2/features"),
                old_feature_receipt_sha256=C.sha256(OLD / "dinov2/features/receipt.json"),
                old_feature_sha256=C.sha256(OLD / "dinov2/features/features.npz"),
                unchanged_old_X_sha256=matrix_digest(X), checkpoint=str(checkpoint),
                checkpoint_files_sha256=weights, primary_configuration=PRIMARY,
                package_versions=package_versions(),
                compared_configurations=dict(heads=["logistic_regression", "linear_svm"],
                                             balance_modes=["none", "sqrt", "balanced"], C=1.0),
                unique_targets=1083, class_counts=C.class_counts(y), folds=3,
                device="cuda", precision="float32", batch_size=8, threads=4,
                old_vectors_reused=True, all_supervised_heads_refitted=True,
                labels_independently_reviewed=False, final_classifier_exported=False,
                external_val_test_read=False, network_required=False,
                interpretation="Paired crop-pipeline comparison: native pixels, context, local brightness statistics and resampling change together; no isolated black-padding causal claim.",
                limitations=["Catalog types unreviewed: FL15, SP94.",
                             "Classifier-internal grouped diagnostics, not independent cascade acceptance.",
                             "No geographic buffer/pretraining-region isolation claim.",
                             "Primary configuration chosen before new-crop scores; secondary six configurations are paired rather than separately optimized."])
    write(out / "protocol.json", spec)
    (out / "protocol.sha256").write_text(C.sha256(out / "protocol.json")+"\n", encoding="utf-8")
    write(out / "queue.json", dict(status="prepared_not_started", created_utc=utc(),
                                  native_status=str(NATIVE_STATUS)))
    (out / "README.md").write_text("""# 现有裁片与原始CTX重裁对照

同一1083个编号、目录类型、完整人工框与固定空间分组，DINOv2本地冻结CUDA FP32特征。
主对照提前固定为LR＋逆频率加权，C=1；另报告相同6组LR/SVM与类别加权，逐组配对，不各取最优制造收益。
旧特征逐值复用，全部监督分类器、标准化及类别权重按当前各折重新拟合。
新图使用原生像素、真实上下文及直接224缩放；这是裁片生成方案比较，不能单独归因于黑边。

系统任务顺序完成原生CPU裁图、来源/哈希核查、冻结GPU特征、两组分类头及配对报告。
关闭客户端和断网可以；电脑需保持开机登录。临时防自动睡眠，结束后解除。
queue.json为主进度，原生裁图status.json按目标更新，新特征native/features/status.json按批更新。
分类类型未独立复核，FL仅15个；不读取外部测试，不导出部署分类器。
所有历史数据和实验保留；没有缺图降级、缩框、换标签或自动删目标。
""", encoding="utf-8")
    print(json.dumps(dict(status="prepared", out=str(out), targets=1083, primary=PRIMARY)), flush=True)


def check(out):
    spec = C.read_json(out / "protocol.json")
    if C.sha256(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise ValueError("Paired comparison protocol changed")
    for source, expected in spec["source_code_sha256"].items():
        if C.sha256(source) != expected:
            raise ValueError("Prepared source code changed: " + source)
    for source, expected in ((spec["master_plan"], spec["master_plan_sha256"]),
                             (spec["native_build_plan"], spec["native_build_plan_sha256"]),
                             (spec["old_plan"], spec["old_plan_sha256"]),
                             (Path(spec["old_feature_dir"]) / "receipt.json", spec["old_feature_receipt_sha256"]),
                             (Path(spec["old_feature_dir"]) / "features.npz", spec["old_feature_sha256"])):
        if C.sha256(source) != expected:
            raise ValueError("Prepared input changed: " + str(source))
    for name, expected in spec["checkpoint_files_sha256"].items():
        if C.sha256(Path(spec["checkpoint"]) / name) != expected:
            raise ValueError("Local encoder weights changed")
    if package_versions() != spec["package_versions"]:
        raise ValueError("Prepared package versions changed")
    Q.verify_master(spec["master_plan"])
    return spec


def rebind_old(plan_path, feature_dir, spec):
    original, X, y, _, receipt = C.load_verified_features(spec["old_plan"], spec["old_feature_dir"])
    target = C.verify_plan(plan_path)
    fields = ("source_id", "class_id", "catalog_class", "crop", "crop_sha256")
    if len(original["rows"]) != len(target["rows"]) or any(any(a[k] != b[k] for k in fields) for a, b in zip(original["rows"], target["rows"])):
        raise ValueError("Rebinding would reassign old vectors to different ordered images or IDs")
    if matrix_digest(X) != spec["unchanged_old_X_sha256"] or not np.array_equal(y, [r["class_id"] for r in target["rows"]]):
        raise ValueError("Old frozen features or ordered labels changed")
    folder = C.new_directory(feature_dir)
    np.savez_compressed(folder / "features.npz", X=X, y=y,
                        source_id=np.array([r["source_id"] for r in target["rows"]]),
                        group=np.array([r["spatial_group_id"] for r in target["rows"]]),
                        fold_id=np.array([r["fold_id"] for r in target["rows"]], dtype=np.int64))
    with np.load(folder / "features.npz", allow_pickle=False) as new:
        if not np.array_equal(X, new["X"]) or not np.array_equal(y, new["y"]):
            raise ValueError("Metadata rebind changed old feature values")
    rebound = deepcopy(receipt)
    rebound.update(plan=str(plan_path), plan_sha256=C.sha256(plan_path),
                   feature_file_sha256=C.sha256(folder / "features.npz"),
                   paired_crop_comparison_rebind=dict(source_feature_sha256=spec["old_feature_sha256"],
                       source_receipt_sha256=spec["old_feature_receipt_sha256"],
                       unchanged_X_sha256=spec["unchanged_old_X_sha256"], new_encoder_inference=False,
                       metadata_changes=["source_protocol", "plan", "spatial_group_id", "fold_id"],
                       original_GPU_feature_extraction_preserved=True, current_rebind_CPU_only=True))
    write(folder / "receipt.json", rebound)
    write(folder / "status.json", dict(status="frozen_features_extracted_no_classifier_fitted",
                                      completed_feature_rows=1083, cached_features_reused=True))
    C.load_verified_features(plan_path, folder)


def ensure_gpu_idle():
    import psutil
    own = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    scripts = {"run_full_pretrained_detector.py", "profile_loader_workers.py", "run_fourclass_feature_comparison.py",
               "run_mars_vit_classification.py", "probe_p2_detector.py", "extend_p2_earlystop.py",
               "run_mars_vit_pipeline.py", "run_p2_pilot_pair.py", "recover_p2_pilot_pair.py", Path(__file__).name}
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if proc.pid not in own and any(Path(str(p)).name in scripts for p in (proc.info["cmdline"] or [])):
                raise RuntimeError("Another experiment is active: " + str(proc.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def summarize(out, old_plan, native_plan, spec):
    from sklearn.metrics import confusion_matrix, f1_score
    old = C.read_json(out / "old/cv/cv_report.json")
    new = C.read_json(out / "native/cv/cv_report.json")
    p0, p1 = C.verify_plan(old_plan), C.verify_plan(native_plan)
    fields = ("source_id", "class_id", "catalog_class", "spatial_group_id", "fold_id")
    if len(p0["rows"]) != 1083 or len(p1["rows"]) != 1083 or any(any(a[k] != b[k] for k in fields) for a, b in zip(p0["rows"], p1["rows"])):
        raise ValueError("Comparison lost paired target/label/fold identity")
    old_by = {(r["head"], r["balance_mode"]): r for r in old["results"]}
    new_by = {(r["head"], r["balance_mode"]): r for r in new["results"]}
    if len(old_by) != 6 or set(old_by) != set(new_by):
        raise ValueError("Classifier configurations are not paired")
    y = np.array([r["class_id"] for r in p0["rows"]])
    comparisons = []
    for key, a in old_by.items():
        b = new_by[key]
        pred0, pred1 = np.asarray(a["source_order_predictions"]), np.asarray(b["source_order_predictions"])
        for result, pred in ((a, pred0), (b, pred1)):
            if pred.shape != y.shape or not np.isin(pred, np.arange(4)).all() or not np.array_equal(confusion_matrix(y, pred, labels=range(4)), result["confusion_matrix"]):
                raise ValueError("Invalid paired confusion matrix/predictions")
            if abs(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)-result["out_of_fold_macro_f1"]) > 1e-12:
                raise ValueError("Invalid paired macro-F1")
        diagnostics = []
        padding = np.array([r.get("padding_fraction", 0.) for r in p0["rows"]], dtype=float)
        boxes = [r.get("target_box_image_px") for r in p0["rows"]]
        if any(v is None or len(v) != 4 or v[2] <= v[0] or v[3] <= v[1] for v in boxes):
            raise ValueError("Aspect diagnostics require all verified original-manifest target boxes")
        aspect = np.array([max(v[2]-v[0], v[3]-v[1])/min(v[2]-v[0], v[3]-v[1]) for v in boxes])
        masks = [("padding_lt10pct", padding < .1), ("padding_10to50pct", (padding >= .1) & (padding < .5)),
                 ("padding_ge50pct", padding >= .5), ("aspect_lt3", aspect < 3), ("aspect_ge3", aspect >= 3)]
        for name, mask in masks:
            count = int(mask.sum())
            if not count:
                continue
            diagnostics.append(dict(group=name, targets=count, class_counts=C.class_counts(y[mask]),
                old_accuracy=float(np.mean(pred0[mask] == y[mask])), native_accuracy=float(np.mean(pred1[mask] == y[mask])),
                wrong_to_right=int(((pred0 != y) & (pred1 == y) & mask).sum()),
                right_to_wrong=int(((pred0 == y) & (pred1 != y) & mask).sum()),
                descriptive_subgroup_only=True))
        comparisons.append(dict(head=key[0], balance_mode=key[1], fixed_C=1.0,
            old_macro_f1=a["out_of_fold_macro_f1"], native_macro_f1=b["out_of_fold_macro_f1"],
            native_minus_old_macro_f1=b["out_of_fold_macro_f1"]-a["out_of_fold_macro_f1"],
            old_per_class=a["per_class"], native_per_class=b["per_class"],
            old_confusion_matrix=a["confusion_matrix"], native_confusion_matrix=b["confusion_matrix"],
            wrong_to_right_ids=[r["source_id"] for i, r in enumerate(p0["rows"]) if pred0[i] != y[i] and pred1[i] == y[i]],
            right_to_wrong_ids=[r["source_id"] for i, r in enumerate(p0["rows"]) if pred0[i] == y[i] and pred1[i] != y[i]],
            diagnostics=diagnostics))
    primary = next(r for r in comparisons if r["head"] == PRIMARY["head"] and r["balance_mode"] == PRIMARY["balance_mode"])
    report = dict(status="completed_paired_native_crop_internal_catalog_comparison", created_utc=utc(),
        protocol_sha256=C.sha256(out / "protocol.json"), unique_targets=1083, configurations_per_arm=6,
        encoder="dinov2", primary_configuration=PRIMARY, primary_result=primary,
        comparisons=comparisons, same_ids_labels_folds=True,
        old_plan_sha256=C.sha256(old_plan), native_plan_sha256=C.sha256(native_plan),
        old_cv_report_sha256=C.sha256(out / "old/cv/cv_report.json"), native_cv_report_sha256=C.sha256(out / "native/cv/cv_report.json"),
        native_manifest_sha256=C.sha256(NATIVE_MANIFEST), old_vectors_unchanged=True,
        old_and_new_heads_refitted=True, final_classifier_exported=False, external_val_test_read=False,
        interpretation=spec["interpretation"], limitations=spec["limitations"])
    write(out / "comparison.json", report)
    lines = ["# 现有裁片／原生CTX重裁对照结果", "", "同一1083个目标和修订后的空间折；DINOv2冻结，配对6种固定分类配置。主配置提前固定为LR＋逆频率加权，C=1。", "",
             "| 配置 | 旧裁片macro-F1 | 原生裁片macro-F1 | 差值（百分点） |", "|---|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['head']}／{r['balance_mode']} | {r['old_macro_f1']:.1%} | {r['native_macro_f1']:.1%} | {r['native_minus_old_macro_f1']*100:+.2f} |")
    lines += ["", "主配置逐类表现：", "", "| 类型 | 旧精确率 | 新精确率 | 旧召回率 | 新召回率 | 旧F1 | 新F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in C.CLASS_NAMES:
        a, b = primary["old_per_class"][name], primary["native_per_class"][name]
        lines.append(f"| {name} | {a['precision']:.1%} | {b['precision']:.1%} | {a['recall']:.1%} | {b['recall']:.1%} | {a['f1-score']:.1%} | {b['f1-score']:.1%} |")
    lines += ["", f"旧错新对{len(primary['wrong_to_right_ids'])}个；旧对新错{len(primary['right_to_wrong_ids'])}个。逐编号、黑色填充分组和长宽比分组见comparison.json。", "",
              "裁片方案同时改变真实上下文、亮度统计和缩放链路；结果不能单独归因于黑边。子组统计为描述性证据。类型未独立复核，FL仅15个，不是最终检测分类验收。"]
    (out / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def run(out):
    attempts = F.prevent_network()
    spec = check(out)
    if C.read_json(out / "queue.json")["status"] != "prepared_not_started":
        raise FileExistsError("Prior paired execution preserved; no implicit restart")
    independence = F.process_independence()
    if not independence["scheduler_owned"]:
        raise RuntimeError("Use the independent Windows system task")
    ensure_gpu_idle()
    kernel = ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    queue = dict(status="running", phase="building_native_crops_cpu", pid=os.getpid(), started_utc=utc(),
                 process_independence=independence, offline=True, temporary_system_awake=True,
                 native_status=str(NATIVE_STATUS), completed_arms=[], current_arm="native_crop_build")
    write(out / "queue.json", queue)
    started = time.monotonic()
    try:
        result = N.build(Path(spec["native_build_plan"]))
        if Path(result["manifest"]).resolve() != NATIVE_MANIFEST.resolve():
            raise ValueError("Native builder returned unexpected manifest")
        check(out)
        queue.update(phase="finalizing_verified_input_plans", current_arm=None)
        write(out / "queue.json", queue)
        finalized = Q.finalize(Path(spec["master_plan"]), NATIVE_MANIFEST, out)
        old_plan, new_plan = Path(finalized["old_plan"]), Path(finalized["native_plan"])
        if old_plan.resolve() != (out / "old/plan.json").resolve() or new_plan.resolve() != (out / "native/plan.json").resolve():
            raise ValueError("Finalized plans must remain in their comparison arms")
        queue.update(phase="refitting_old_crop_classifiers", current_arm="old")
        write(out / "queue.json", queue)
        rebind_old(old_plan, out / "old/features", spec)
        C.grouped_cv(old_plan, out / "old/features", out / "old/cv")
        queue["completed_arms"].append("old")
        queue.update(phase="extracting_native_features_cuda", current_arm="native")
        write(out / "queue.json", queue)
        ensure_gpu_idle()
        receipt = C.extract(new_plan, out / "native/features", encoder="dinov2", checkpoint=spec["checkpoint"],
                            batch_size=spec["batch_size"], threads=spec["threads"], device="cuda")
        if receipt["shape"] != [1083, 768] or not receipt["gpu_used"]:
            raise RuntimeError("Native DINOv2 extraction did not produce all CUDA feature rows")
        queue.update(phase="refitting_native_crop_classifiers")
        write(out / "queue.json", queue)
        C.grouped_cv(new_plan, out / "native/features", out / "native/cv")
        queue["completed_arms"].append("native")
        report = summarize(out, old_plan, new_plan, spec)
        queue.update(status=report["status"], phase="complete", current_arm=None)
    except BaseException as error:
        queue.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(), elapsed_seconds=round(time.monotonic()-started, 3),
                     temporary_system_awake=False, blocked_network_attempts=attempts)
        write(out / "queue.json", queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "scheduled-run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "runs/experiments"):
        raise ValueError("Use a new comparison inside this workspace")
    if args.mode == "prepare":
        prepare(out)
    else:
        sys.stdout = (out / "training.log").open("x", encoding="utf-8", buffering=1)
        sys.stderr = (out / "training.err.log").open("x", encoding="utf-8", buffering=1)
        run(out)


if __name__ == "__main__":
    main()
