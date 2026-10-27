"""Independent offline DINOv2/ResNet18 grouped classifier screening."""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if "scheduled-run" in sys.argv:
    sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages"))
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")

import extract_manual_classification_features as C
import run_full_pretrained_detector as F

OUT = ROOT / "runs/experiments/fourclass_features_20261015_v1"
SOURCE_PLAN = ROOT / "runs/research/classification_next_20261014_v1/plan.json"
PREDECESSOR = ROOT / "runs/experiments/loader_workers_profile_20261014_v1/queue.json"
WEIGHT_RECEIPT = ROOT / "runs/research/classification_weight_sources_20261015_v1/resnet18_receipt.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(out):
    if out.exists():
        raise FileExistsError("Preserve prior classifier experiment")
    old = C.verify_plan(SOURCE_PLAN)
    source_receipt = C.read_json(WEIGHT_RECEIPT)
    resnet = Path(source_receipt["path"])
    if source_receipt["url"] != "https://download.pytorch.org/models/resnet18-f37072fd.pth" or not source_receipt["sha256"].startswith("f37072fd"):
        raise ValueError("ResNet18 origin is not the declared official ImageNet V1 weight")
    if C.sha256(resnet) != source_receipt["sha256"]:
        raise ValueError("ResNet18 downloaded bytes changed")
    dino = C.cached_dino_snapshot()
    if not dino:
        raise ValueError("DINOv2 local weights unavailable")
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    copied = out / "inputs/resnet18-f37072fd.pth"
    shutil.copy2(resnet, copied)
    shutil.copy2(WEIGHT_RECEIPT, out / "inputs/resnet18_source_receipt.json")
    plan = deepcopy(old)
    plan["created_utc"] = utc()
    plan["status"] = "locked_before_cuda_features_and_heads"
    plan["original_prepared_plan"] = str(SOURCE_PLAN)
    plan["original_prepared_plan_sha256"] = C.sha256(SOURCE_PLAN)
    plan["preprocessing"]["encoder_mode"] = "eval, requires_grad=False, CUDA, float32, inference_mode; autocast/TF32 disabled"
    plan["resource_receipt"] = dict(gpu_requested=True, device="cuda", batch_size=8, threads=4,
        training_started=False, source_manifest_modified=False, original_folds_preserved=True)
    write(out / "plan.json", plan)
    encoders = [dict(name="dinov2", checkpoint=dino, dimension=768,
                    files={name: C.sha256(Path(dino) / name) for name in ("config.json", "model.safetensors")}),
                dict(name="resnet18", checkpoint=str(copied), dimension=512,
                    files={copied.name: C.sha256(copied)})]
    protocol = dict(created_utc=utc(), status="locked_before_execution", scope=C.POLICY,
        original_plan_sha256=C.sha256(SOURCE_PLAN), plan_sha256=C.sha256(out / "plan.json"),
        script_sha256=C.sha256(__file__), extractor_sha256=C.sha256(C.__file__),
        offline_helper_sha256=C.sha256(F.__file__), source_weight_receipt_sha256=C.sha256(out / "inputs/resnet18_source_receipt.json"),
        encoders=encoders, device="cuda", precision="float32", batch_size=8, threads=4,
        prior_speed_profile=str(PREDECESSOR), wait_for_prior_speed_profile=True,
        unique_source_count=1083, folds=3, subtype_labels_independently_reviewed=False,
        final_classifier_exported=False, external_val_test_access=False,
        heads=plan["cv_comparison"]["heads"], balance_modes=plan["cv_comparison"]["balance_modes"],
        mars_vit=dict(status="pending_separate_weight_architecture_verification", enrolled=False,
            source="https://zenodo.org/records/18180801", note="Do not silently substitute Base for the planned Small arm."),
        limitations=["Catalog subtype labels remain unreviewed; FL has only 15 unique IDs.",
                     "Internal tile-connected folds do not prove adjacent-context geographic buffer separation.",
                     "Different encoders/pretraining are screening comparisons, not causal isolation of pretraining domain.",
                     "No final classifier exported; all scores are internal out-of-fold diagnostics."])
    write(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(C.sha256(out / "protocol.json") + "\n", encoding="utf-8")
    write(out / "state.json", dict(status="prepared_not_started", created_utc=utc()))
    (out / "README.md").write_text("""# 人工裁片四分类特征与分类器对照

同一1083个独立滑坡编号、原先冻结的3折空间分组；没有重新划分或打开外部验证/测试图。
DINOv2-base与ImageNet ResNet18分别冻结提取CUDA FP32特征，使用相同224完整裁片和ImageNet归一化。
每种特征比较LR/线性SVM × 不加权/平方根加权/逆频率加权，共12组；标准化和类别权重只拟合各折训练数据。

类型仍来自未独立复核目录，FL只有15个独立目标，结果是内部方向筛选，不能作为最终四分类验收。
Mars-ViT Small待权重架构核验，本轮不把Base替换进来。

系统后台任务独立于客户端，先等待当前速度短测完成再使用GPU。全部权重预先保存在本机；运行时禁止联网。
状态为queue.json；每批特征进度见各编码器features/status.json；完成后生成comparison.json和results.md。
锁屏、屏幕关闭和断网可以；电脑需要保持开机登录。异常或中断不会覆盖已有输出，也不会自动换CPU或放宽分组。
""", encoding="utf-8")
    print("PREPARED", out, flush=True)


def check(out):
    spec = C.read_json(out / "protocol.json")
    if C.sha256(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol digest changed")
    for path, key in ((Path(__file__), "script_sha256"), (Path(C.__file__), "extractor_sha256"),
                      (Path(F.__file__), "offline_helper_sha256"), (out / "plan.json", "plan_sha256"),
                      (out / "inputs/resnet18_source_receipt.json", "source_weight_receipt_sha256")):
        if C.sha256(path) != spec[key]:
            raise ValueError("Prepared code/plan receipt changed: " + str(path))
    for encoder in spec["encoders"]:
        base = Path(encoder["checkpoint"])
        for name, digest in encoder["files"].items():
            if C.sha256(base / name if base.is_dir() else base) != digest:
                raise ValueError("Preloaded encoder weight changed")
    C.verify_plan(out / "plan.json")
    return spec


def run(out):
    import psutil
    attempts = F.prevent_network()
    spec = check(out)
    if (out / "queue.json").exists():
        raise FileExistsError("Preserve existing execution; no silent restart")
    independence = F.process_independence()
    if not independence["scheduler_owned"]:
        raise RuntimeError("Use the independent system task")
    kernel = ctypes.windll.kernel32
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    queue = dict(status="waiting_for_speed_profile", pid=os.getpid(), started_utc=utc(),
        process_independence=independence, offline=True, temporary_system_awake=True,
        current_encoder=None, completed_encoders=[], predecessor=str(PREDECESSOR))
    write(out / "queue.json", queue)
    try:
        while True:
            previous = C.read_json(PREDECESSOR)
            if previous["status"] == "completed_engineering_only":
                break
            if previous["status"] != "running" or not psutil.pid_exists(previous["pid"]):
                raise RuntimeError("Preceding speed profile did not finish normally; inspect it before using GPU")
            time.sleep(15)
        queue.update(status="running", predecessor_finished_utc=previous.get("finished_utc"), activated_utc=utc())
        write(out / "queue.json", queue)
        write(out / "state.json", dict(status="running", activated_utc=queue["activated_utc"]))
        reports = []
        for encoder in spec["encoders"]:
            name = encoder["name"]
            queue.update(current_encoder=name, phase="extracting_features")
            write(out / "queue.json", queue)
            folder = out / name
            folder.mkdir()
            receipt = C.extract(out / "plan.json", folder / "features", encoder=name,
                checkpoint=encoder["checkpoint"], batch_size=spec["batch_size"], threads=spec["threads"], device=spec["device"])
            if receipt["shape"] != [1083, encoder["dimension"]] or not receipt["gpu_used"]:
                raise RuntimeError("Missing full GPU frozen feature matrix")
            queue["phase"] = "fitting_grouped_classifiers"
            write(out / "queue.json", queue)
            C.grouped_cv(out / "plan.json", folder / "features", folder / "cv")
            report = C.read_json(folder / "cv/cv_report.json")
            if len(report["results"]) != 6 or report["external_val_test_read"] or report["final_classifier_exported"]:
                raise RuntimeError("Unexpected classifier screening scope")
            reports.append(dict(encoder=name, feature_receipt_sha256=C.sha256(folder / "features/receipt.json"),
                cv_report_sha256=C.sha256(folder / "cv/cv_report.json"), results=report["results"]))
            queue["completed_encoders"].append(name)
            write(out / "queue.json", queue)
        comparison = dict(status="completed_internal_catalog_comparison", created_utc=utc(),
            protocol_sha256=C.sha256(out / "protocol.json"), plan_sha256=spec["plan_sha256"],
            unique_source_count=1083, folds=3, final_classifier_exported=False, external_val_test_read=False,
            encoders=reports, limitations=spec["limitations"], blocked_network_attempts=attempts)
        write(out / "comparison.json", comparison)
        lines = ["# 四分类内部对照结果", "", "同一1083个独立编号与冻结3折，CUDA FP32冻结特征；不读取外部验证/测试图片。", "",
            "| 特征 | 分类器 | 类别加权 | macro-F1 | SL召回 | SP召回 | FL召回 | CL召回 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for report in reports:
            for result in report["results"]:
                classes = result["per_class"]
                recalls = " | ".join(f"{classes[c]['recall']:.1%}" for c in C.CLASS_NAMES)
                lines.append(f"| {report['encoder']} | {result['head']} | {result['balance_mode']} | {result['out_of_fold_macro_f1']:.1%} | {recalls} |")
        lines += ["", "查看各模型cv_report.json中的逐类精确率/F1、混淆矩阵、每折和空间组表现。", "",
            "分类目录标签尚未单独复核，FL只有15个独立编号；这是方向筛选，不能当独立验收或完整检测分类流程成绩。尚未导出最终部署分类器。"]
        (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        queue.update(status="completed_internal_catalog_comparison", current_encoder=None, phase="complete")
    except BaseException as error:
        queue.update(status="failed", error=repr(error))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        queue.update(finished_utc=utc(), temporary_system_awake=False, blocked_network_attempts=attempts)
        write(out / "queue.json", queue)
        write(out / "state.json", dict(status=queue["status"], finished_utc=queue["finished_utc"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "scheduled-run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "runs/experiments"):
        raise ValueError("Experiment must remain in project")
    if args.mode == "prepare":
        prepare(out)
    else:
        sys.stdout = (out / "training.log").open("x", encoding="utf-8", buffering=1)
        sys.stderr = (out / "training.err.log").open("x", encoding="utf-8", buffering=1)
        run(out)


if __name__ == "__main__":
    main()
