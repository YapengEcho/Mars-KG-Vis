"""Independent offline author-configured Mars-ViT Base four-class screening.

This is a separately named Base arm, not a substitution for the unverified Small
arm. Preparation loads the local checkpoint strictly on CPU, but does not run
inference. The system-scheduled action extracts frozen CUDA FP32 CLS features and
reuses the existing grouped classifier protocol without changing its data/folds.
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
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")

import numpy as np

import extract_manual_classification_features as C
import run_full_pretrained_detector as F

OUT = ROOT / "runs/experiments/mars_vit_base_fourclass_20261015_v2"
SOURCE_PLAN = ROOT / "runs/research/classification_pixel_provenance_20261015_v2/plan.json"
WEIGHT = ROOT / "runs/research/mars_vit_weights_20261015_v1/mars-mae-dino-vit-base-v1.pth"
SOURCE_RECEIPT = WEIGHT.parent / "receipt.json"
OFFICIAL_SIZE = 341679467
OFFICIAL_MD5 = "a2b6c983546329a269f3e512dc2a762e"
OFFICIAL_FILE = "https://zenodo.org/records/18180801/files/mars-mae-dino-vit-base-v1.pth"
MODEL_NAME = "vit_base_patch16_224"
ENCODER = "mars_vit_base"
AUTHOR_MODEL = dict(pretrained=False, in_chans=1, num_classes=0, global_pool="")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    C.write_json_atomic(path, value)


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "timm", "pillow", "numpy", "scikit-learn")}


def verify_weight(weight, receipt_path):
    weight, receipt_path = Path(weight).resolve(), Path(receipt_path).resolve()
    source = C.read_json(receipt_path)
    if not weight.is_file() or weight.stat().st_size != OFFICIAL_SIZE:
        raise ValueError("Mars-ViT Base local file does not match official file size")
    md5, sha = file_digest(weight, "md5"), C.sha256(weight)
    if md5 != OFFICIAL_MD5:
        raise ValueError("Mars-ViT Base bytes do not match official Zenodo MD5")
    if Path(source["path"]).resolve() != weight or source["sha256"] != sha:
        raise ValueError("Mars-ViT Base local source receipt path/SHA256 differs")
    if source["md5"] != md5 or int(source["bytes"]) != OFFICIAL_SIZE:
        raise ValueError("Mars-ViT Base local source receipt MD5/size differs")
    if source["url"].split("?", 1)[0] != OFFICIAL_FILE:
        raise ValueError("Mars-ViT Base source receipt is not the declared author file")
    return dict(path=str(weight), filename=weight.name, bytes=OFFICIAL_SIZE,
                md5=md5, sha256=sha, official_file=OFFICIAL_FILE,
                source_receipt_path=str(receipt_path),
                source_receipt_sha256=C.sha256(receipt_path), source_receipt=source)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def strict_encoder(weight):
    """No key stripping, shape remapping, omitted head, or partial transfer."""
    import torch
    import timm

    model = timm.create_model(MODEL_NAME, **AUTHOR_MODEL).cpu().float()
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    wrapper = None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # The released backbone file uses timm's documented state_dict wrapper.
        # Accept the verified author's exact wrapper, not generic teacher/model
        # selection or arbitrary nested payloads, and retain every metadata field.
        expected_wrapper = {"state_dict", "timm_name", "arch", "img_size", "in_chans", "num_classes", "source"}
        if set(checkpoint) != expected_wrapper:
            raise ValueError("Author checkpoint wrapper fields changed: " + str(sorted(checkpoint)))
        expected_metadata = dict(timm_name=MODEL_NAME, arch="vit_base", img_size=224,
                                 in_chans=1, num_classes=0)
        if any(checkpoint[k] != v for k, v in expected_metadata.items()) or not isinstance(checkpoint["source"], str):
            raise ValueError("Author checkpoint wrapper metadata conflicts with the Base architecture")
        wrapper = dict(payload_key="state_dict", metadata={k: checkpoint[k] for k in sorted(checkpoint) if k != "state_dict"},
                       wrapper_fields=sorted(checkpoint), tensor_payload_complete_required=True)
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected the author's entire tensor dictionary or exact published state_dict wrapper")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatch = {k: dict(weight=list(state[k].shape), model=list(expected[k].shape))
                for k in set(expected) & set(state) if state[k].shape != expected[k].shape}
    if missing or unexpected or mismatch:
        raise ValueError("Strict author architecture mismatch: " + json.dumps(
            dict(missing=missing, unexpected=unexpected, shape_mismatch=mismatch)))
    if any(v.dtype != expected[k].dtype for k, v in state.items()):
        raise ValueError("Checkpoint dtype differs; implicit tensor casting is prohibited")
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Strict load unexpectedly omitted checkpoint tensors")
    source_state_hash, loaded_state_hash = state_hash(state), state_hash(model.state_dict())
    if source_state_hash != loaded_state_hash:
        raise RuntimeError("Loaded model state differs from entire author checkpoint")
    if model.num_features != 768 or model.patch_embed.proj.in_channels != 1:
        raise ValueError("Expected Mars-ViT Base grayscale 768-dimensional encoder")
    if tuple(model.patch_embed.img_size) != (224, 224) or tuple(model.patch_embed.patch_size) != (16, 16):
        raise ValueError("Expected the author 224 input and patch16 model")
    model.eval().requires_grad_(False)
    receipt = dict(encoder=ENCODER, author_architecture=MODEL_NAME,
        architecture_arguments=AUTHOR_MODEL, source_state_sha256=source_state_hash,
        loaded_state_sha256=loaded_state_hash, strict=True, tensor_count=len(state),
        parameters=sum(p.numel() for p in model.parameters()), missing_keys=[],
        unexpected_keys=[], shape_remapping=False, key_remapping=False,
        source_checkpoint_wrapping=wrapper, partial_pretraining=False,
        features=768, feature_endpoint="forward_features(pixels)[:, 0] CLS token",
        package_versions=package_versions(), mode="eval", requires_grad=False,
        parameter_dtype="float32", checkpoint=str(Path(weight).resolve()))
    return model, receipt


def prepare(out, weight=WEIGHT, source_receipt=SOURCE_RECEIPT):
    """Freeze a separate plan and prove strict author-weight compatibility on CPU."""
    # The earlier classification folds missed donor tile provenance. Never create
    # a new experiment until the corrected frozen plan is available and valid.
    if not SOURCE_PLAN.is_file():
        raise FileNotFoundError("Wait for corrected donor-tile pixel-provenance plan: " + str(SOURCE_PLAN))
    original = C.verify_plan(SOURCE_PLAN)
    out = C.new_directory(out)
    status = dict(status="preparing_cpu_metadata_only", started_utc=utc(),
        gpu_used=False, inference_started=False, classifier_training_started=False)
    write(out / "preparation_status.json", status)
    try:
        attempts = F.prevent_network()
        if original["unique_source_count"] != 1083 or len(original["rows"]) != 1083 or original["folds"] != 3:
            raise ValueError("Require the corrected 1083-target, 3-fold pixel-provenance classification plan")
        if original["cv_comparison"]["heads"] != ["logistic_regression", "linear_svm"] or original["cv_comparison"]["balance_modes"] != ["none", "sqrt", "balanced"] or original["cv_comparison"]["C"] != 1.0:
            raise ValueError("Existing fixed six-configuration protocol changed")
        weight_receipt = verify_weight(weight, source_receipt)
        import torch
        torch.set_num_threads(4)
        model, architecture = strict_encoder(weight)
        del model
        plan = deepcopy(original)
        plan.update(created_utc=utc(), status="locked_before_mars_vit_base_cuda_features_and_heads",
            original_prepared_plan=str(SOURCE_PLAN), original_prepared_plan_sha256=C.sha256(SOURCE_PLAN))
        plan["preprocessing"] = dict(
            encoder_mode="eval, requires_grad=False, CUDA float32 inference_mode; autocast/TF32 disabled",
            input="Full square crop converted to grayscale L, bilinear direct resize to 224; no center crop or augmentation",
            normalization=dict(mean=[0.5], std=[0.25]), dimension="Mars-ViT Base CLS=768",
            fold_transform=original["preprocessing"]["fold_transform"],
            author_preprocessing=True,
            difference_from_previous_arms="Author grayscale/normalization and model/pretraining differ from RGB ImageNet-normalized DINOv2/ResNet18; this is screening, not isolated domain-pretraining causality")
        plan["resource_receipt"] = dict(gpu_requested=True, device="cuda", batch_size=8,
            threads=4, training_started=False, source_manifest_modified=False, original_folds_preserved=True)
        write(out / "plan.json", plan)
        write(out / "weight_verification.json", weight_receipt)
        write(out / "architecture_verification.json", architecture)
        protocol = dict(created_utc=utc(), status="locked_before_execution", scope=C.POLICY,
            arm=ENCODER, explicitly_separate_base_arm=True,
            planned_small_arm=dict(status="released_checkpoint_constructor_mapping_not_verified",
                                  author_constructor_available=True, replaced=False),
            source_plan=str(SOURCE_PLAN), source_plan_sha256=C.sha256(SOURCE_PLAN),
            plan_sha256=C.sha256(out / "plan.json"), script_sha256=C.sha256(__file__),
            extractor_sha256=C.sha256(C.__file__), offline_helper_sha256=C.sha256(F.__file__),
            architecture_receipt_sha256=C.sha256(out / "architecture_verification.json"),
            weight_verification_sha256=C.sha256(out / "weight_verification.json"),
            weight=weight_receipt, package_versions=architecture["package_versions"],
            device="cuda", precision="float32", batch_size=8, threads=4,
            features=768, unique_source_count=1083, folds=3, class_mapping=C.CLASS_TO_ID,
            original_rows_and_fold_assignments_preserved=True,
            subtype_labels_independently_reviewed=False, final_classifier_exported=False,
            external_val_test_access=False, heads=plan["cv_comparison"]["heads"],
            balance_modes=plan["cv_comparison"]["balance_modes"], C=1.0,
            preparation_cpu_only=True, preparation_blocked_network_attempts=attempts,
            limitations=["Catalog labels unreviewed; FL has 15 independent target IDs.",
                "Author grayscale normalization differs from the two RGB encoder arms.",
                "Model architecture and pretraining both differ: no isolated domain advantage claim.",
                "Internal tile-connected folds do not establish adjacent-region buffer separation.",
                "Base is a separate additional arm; author Small constructor exists but its released checkpoint mapping is not yet verified.",
                "No independent final test or final deployment classifier is produced."])
        write(out / "protocol.json", protocol)
        (out / "protocol.sha256").write_text(C.sha256(out / "protocol.json") + "\n", encoding="utf-8")
        (out / "README.md").write_text("""# Mars-ViT Base 四分类独立对照

这是新增的 Base 组；作者 Small 构造函数存在，但与发布权重的对应关系尚未核验，原计划和已有结果均保留。
作者公开权重采用 timm vit_base_patch16_224、单通道、num_classes=0、global_pool=''。
发布文件用 state_dict 包裹完整骨干，包装字段及架构元数据也记录并核对。
完整张量字典严格加载，禁止省略键、部分迁移、修改键名或尺寸。prepare 只用CPU验证模型元数据，不推理。

使用纠正像素来源后的1083个独立编号、冻结3折及原类别映射；与重新计算的DINOv2/ResNet18组比较。
旧组的划分未纳入跨图幅裁片的全部供图图幅，保留为旧探索结果，不能与本组直接比较。
作者输入为224完整灰度裁片，均值0.5、标准差0.25；模型和预处理均不同，结果是方向筛选，不能单独归因于火星预训练。
冻结模型在CUDA FP32下提取768维CLS；无AMP、TF32、增强或CPU回退。
复用原LR/线性SVM × 不加权/平方根/逆频率，共6组，C固定1，标准化和类别权重只拟合各折训练数据。

目录类型尚未独立审核，FL仅15个编号，不访问外部验证/测试数据，也不导出最终分类器。
系统计划任务执行，运行时禁止联网，临时防止自动睡眠，退出后恢复；电脑需保持开机登录。
queue.json、features/status.json按批原子更新；完成后results.md、cv/cv_report.json可查看结果。
失败或中断保留记录；本脚本禁止覆盖已有实验或自动重启。
""", encoding="utf-8")
        status.update(status="prepared_strict_cpu_verified_no_inference", finished_utc=utc())
        write(out / "state.json", dict(status="prepared_not_started", created_utc=utc()))
        print("PREPARED", out, flush=True)
        return protocol
    except BaseException as error:
        status.update(status="preparation_failed", error_type=type(error).__name__, error=str(error), finished_utc=utc())
        raise
    finally:
        write(out / "preparation_status.json", status)


def check(out):
    spec = C.read_json(out / "protocol.json")
    if C.sha256(out / "protocol.json") != (out / "protocol.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("Prepared protocol digest changed")
    for path, key in ((Path(__file__), "script_sha256"), (Path(C.__file__), "extractor_sha256"),
            (Path(F.__file__), "offline_helper_sha256"), (out / "plan.json", "plan_sha256"),
            (Path(spec["source_plan"]), "source_plan_sha256"),
            (out / "architecture_verification.json", "architecture_receipt_sha256"),
            (out / "weight_verification.json", "weight_verification_sha256")):
        if C.sha256(path) != spec[key]:
            raise ValueError("Prepared input/code receipt changed: " + str(path))
    if package_versions() != spec["package_versions"]:
        raise ValueError("Prepared dependency versions changed")
    weight = verify_weight(spec["weight"]["path"], spec["weight"]["source_receipt_path"])
    if weight != spec["weight"]:
        raise ValueError("Local author-weight provenance changed")
    plan = C.verify_plan(out / "plan.json")
    original = C.verify_plan(SOURCE_PLAN)
    if plan["rows"] != original["rows"] or plan["fold_summaries"] != original["fold_summaries"] or plan["cv_comparison"] != original["cv_comparison"]:
        raise ValueError("Source rows, folds or classifier protocol were modified")
    return spec, plan


def ensure_gpu_idle():
    import psutil
    own = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    runners = {"run_full_pretrained_detector.py", "profile_loader_workers.py",
        "run_fourclass_feature_comparison.py", "probe_p2_detector.py",
        "extend_p2_earlystop.py", "run_p2_pilot_pair.py", Path(__file__).name}
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if proc.pid not in own and any(Path(str(x)).name in runners for x in (proc.info["cmdline"] or [])):
                raise RuntimeError(f"Another declared GPU experiment is active: {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def grayscale_pixels(images):
    import torch
    from PIL import Image
    gray = [np.asarray(im.convert("L").resize((224, 224), Image.Resampling.BILINEAR),
                       dtype=np.uint8) for im in images]
    pixels = torch.from_numpy(np.stack(gray)).unsqueeze(1).float() / 255
    return (pixels - 0.5) / 0.25


def extract(out, spec, plan, queue):
    import torch
    from PIL import Image

    feature_out = C.new_directory(out / "features")
    started = time.monotonic()
    progress = dict(status="verifying_crops", encoder=ENCODER, scope=C.POLICY,
        started_utc=utc(), plan_sha256=spec["plan_sha256"],
        completed_feature_rows=0, verified_crop_rows=0, total=len(plan["rows"]),
        requested_device="cuda", downloads=False, classifier_training_started=False,
        batch_size=spec["batch_size"], threads=spec["threads"])

    def publish(status):
        progress.update(status=status, updated_utc=utc(), elapsed_seconds=round(time.monotonic() - started, 3))
        write(feature_out / "status.json", progress)
        queue.update(phase=status, completed_feature_rows=progress["completed_feature_rows"],
            verified_crop_rows=progress["verified_crop_rows"], updated_utc=progress["updated_utc"])
        write(out / "queue.json", queue)

    try:
        publish("verifying_crops")
        for index, row in enumerate(plan["rows"]):
            if C.sha256(row["crop"]) != row["crop_sha256"]:
                raise ValueError("Changed frozen crop: " + row["source_id"])
            progress["verified_crop_rows"] = index + 1
            if (index + 1) % 100 == 0:
                publish("verifying_crops")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this arm does not fall back to CPU")
        torch.set_num_threads(spec["threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        publish("loading_strict_local_encoder")
        model, receipt = strict_encoder(spec["weight"]["path"])
        verified = C.read_json(out / "architecture_verification.json")
        if receipt != verified:
            raise ValueError("Strict runtime CPU model metadata changed from preparation")
        device = torch.device("cuda", torch.cuda.current_device())
        model.to(device).float().eval().requires_grad_(False)
        if any(p.requires_grad or p.device != device or p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("Frozen CUDA FP32 encoder invariants failed")
        if any(b.device != device or (b.is_floating_point() and b.dtype != torch.float32) for b in model.buffers()):
            raise RuntimeError("Encoder buffer precision/device invariant failed")
        chunks = []
        publish("extracting")
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for start in range(0, len(plan["rows"]), spec["batch_size"]):
                rows = plan["rows"][start:start + spec["batch_size"]]
                images = []
                try:
                    for row in rows:
                        with Image.open(row["crop"]) as im:
                            if im.width != im.height:
                                raise ValueError("Frozen subtype crop is not square")
                            images.append(im.convert("L"))
                    tokens = model.forward_features(grayscale_pixels(images).to(device, dtype=torch.float32))
                    if tokens.shape != (len(rows), 197, 768):
                        raise RuntimeError("Unexpected Base patch16/224 CLS token layout")
                    features = tokens[:, 0]
                    if features.dtype != torch.float32 or features.device != device:
                        raise RuntimeError("Extracted features changed declared device/precision")
                    vector = features.cpu().numpy().astype(np.float32)
                finally:
                    for im in images:
                        im.close()
                if vector.shape != (len(rows), 768) or not np.isfinite(vector).all():
                    raise RuntimeError("Unexpected or non-finite features")
                chunks.append(vector)
                progress["completed_feature_rows"] = start + len(rows)
                publish("extracting")
                print(json.dumps(progress), flush=True)
        X = np.concatenate(chunks)
        publish("saving_verified_features")
        temporary = feature_out / "features.incomplete.npz"
        np.savez_compressed(temporary, X=X,
            y=np.array([r["class_id"] for r in plan["rows"]], dtype=np.int64),
            source_id=np.array([r["source_id"] for r in plan["rows"]]),
            group=np.array([r["spatial_group_id"] for r in plan["rows"]]),
            fold_id=np.array([r["fold_id"] for r in plan["rows"]], dtype=np.int64))
        temporary.replace(feature_out / "features.npz")
        receipt.update(status="frozen_features_extracted_no_classifier_fitted", scope=C.POLICY,
            plan=str(out / "plan.json"), plan_sha256=spec["plan_sha256"],
            extractor_script_sha256=spec["script_sha256"],
            reused_classifier_code_sha256=spec["extractor_sha256"],
            checkpoint_files_sha256={Path(spec["weight"]["path"]).name: spec["weight"]["sha256"]},
            weight_verification_sha256=spec["weight_verification_sha256"],
            preprocessing=plan["preprocessing"], requested_device="cuda", device=str(device),
            output_dtype="float32", autocast_enabled=False, tf32_enabled=False,
            gpu_used=True, downloads=False, cpu_only=False,
            cuda_device_name=torch.cuda.get_device_name(device), cuda_runtime=torch.version.cuda,
            feature_file_sha256=C.sha256(feature_out / "features.npz"), shape=list(X.shape),
            torch_version=str(torch.__version__), threads=spec["threads"], batch_size=spec["batch_size"],
            elapsed_seconds=round(time.monotonic() - started, 3), class_mapping=C.CLASS_TO_ID,
            source_ids=[r["source_id"] for r in plan["rows"]],
            subtype_labels_independently_reviewed=False, classifier_training_started=False)
        write(feature_out / "receipt.json", receipt)
        publish("frozen_features_extracted_no_classifier_fitted")
        return receipt
    except BaseException as error:
        progress.update(error_type=type(error).__name__, error=str(error), output_is_complete=False)
        publish("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        raise


def run(out):
    if (out / "queue.json").exists():
        raise FileExistsError("Prior runtime preserved; no silent restart or overwrite")
    queue = dict(status="starting", pid=os.getpid(), started_utc=utc(), encoder=ENCODER,
        scope=C.POLICY, phase="checking_locked_inputs", offline=True,
        temporary_system_awake=False, completed_feature_rows=0, total_feature_rows=1083)
    write(out / "queue.json", queue)
    attempts, awake = [], False
    kernel = ctypes.windll.kernel32
    try:
        attempts = F.prevent_network()
        spec, plan = check(out)
        independence = F.process_independence()
        queue["process_independence"] = independence
        if not independence["scheduler_owned"]:
            raise RuntimeError("Use the independent Windows system task")
        ensure_gpu_idle()
        if not kernel.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError()
        awake = True
        queue.update(status="running", temporary_system_awake=True, activated_utc=utc())
        write(out / "queue.json", queue)
        write(out / "state.json", dict(status="running", activated_utc=queue["activated_utc"]))
        receipt = extract(out, spec, plan, queue)
        if receipt["shape"] != [1083, 768]:
            raise RuntimeError("Missing complete Base feature matrix")
        queue["phase"] = "fitting_original_grouped_classifiers"
        write(out / "queue.json", queue)
        C.grouped_cv(out / "plan.json", out / "features", out / "cv")
        report = C.read_json(out / "cv/cv_report.json")
        if len(report["results"]) != 6 or report["final_classifier_exported"] or report["external_val_test_read"]:
            raise RuntimeError("Classifier result differs from fixed screening scope")
        comparison = dict(status="completed_internal_catalog_comparison", created_utc=utc(),
            scope=C.POLICY, protocol_sha256=C.sha256(out / "protocol.json"), plan_sha256=spec["plan_sha256"],
            feature_receipt_sha256=C.sha256(out / "features/receipt.json"),
            cv_report_sha256=C.sha256(out / "cv/cv_report.json"), unique_source_count=1083,
            folds=3, encoder=ENCODER, explicitly_separate_base_arm=True,
            original_folds_preserved=True, final_classifier_exported=False, external_val_test_read=False,
            results=report["results"], limitations=spec["limitations"], blocked_network_attempts=attempts)
        write(out / "comparison.json", comparison)
        lines = ["# Mars-ViT Base 内部四分类结果", "",
            "相同1083个编号与冻结3折；作者灰度预处理，CUDA FP32冻结768维CLS。", "",
            "| 分类器 | 加权 | macro-F1 | SL召回 | SP召回 | FL召回 | CL召回 |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for result in report["results"]:
            recalls = " | ".join(f"{result['per_class'][c]['recall']:.1%}" for c in C.CLASS_NAMES)
            lines.append(f"| {result['head']} | {result['balance_mode']} | {result['out_of_fold_macro_f1']:.1%} | {recalls} |")
        lines += ["", "这是新增Base组，未替换计划的Small。灰度处理、模型架构及预训练均不同，只能作为方向筛选。",
            "目录类型尚未独立核验，FL只有15个目标；不能当最终验收。逐类精确率/F1、混淆矩阵及每折见cv/cv_report.json。"]
        (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        queue.update(status="completed_internal_catalog_comparison", phase="complete", classifier_configurations=6)
    except BaseException as error:
        queue.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if awake:
            kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(), temporary_system_awake=False, blocked_network_attempts=attempts)
        write(out / "queue.json", queue)
        write(out / "state.json", dict(status=queue["status"], finished_utc=queue["finished_utc"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "scheduled-run"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--weight", type=Path, default=WEIGHT)
    parser.add_argument("--source-receipt", type=Path, default=SOURCE_RECEIPT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "runs/experiments"):
        raise ValueError("Experiment must remain inside project runs/experiments")
    if args.mode == "prepare":
        prepare(out, args.weight.resolve(), args.source_receipt.resolve())
    else:
        sys.stdout = (out / "training.log").open("x", encoding="utf-8", buffering=1)
        sys.stderr = (out / "training.err.log").open("x", encoding="utf-8", buffering=1)
        run(out)


if __name__ == "__main__":
    main()
