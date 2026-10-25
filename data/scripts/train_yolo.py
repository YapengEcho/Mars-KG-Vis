# -*- coding: utf-8 -*-
"""跑 YOLOv8n 检测基线训练（SHP 真框驱动数据集，空间隔离划分已验证）。

环境（RTX 5060 Laptop 8GB，Blackwell 硬要求 cu128）：
    .venv\\Scripts\\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    .venv\\Scripts\\python.exe -m pip install ultralytics

用法：
    python data/scripts/train_yolo.py                      # 用默认数据目录训练
    python data/scripts/train_yolo.py --data <data.yaml> --epochs 150 --imgsz 1024

说明：
  - 读取 make_yolo_dataset.py 生成的 data.yaml（含 train/val/test 三集划分）。
  - 多尺度增强 scale=0.5 提升小滑坡（SP/CL）检测。
  - 训练后自动跑 val，输出按类 P/R/mAP50/mAP50-95。
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROJECT = BASE / "runs" / "detect"


def _build_run_note(note: str | None, ignore_manifest: str | None) -> str | None:
    """Add an immutable sidecar fingerprint to the run report note."""
    parts = [note] if note else []
    if ignore_manifest:
        path = Path(ignore_manifest)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts.append(
            "ignore_sidecar="
            f"{path.as_posix()} sha256={digest} ignored_rows={len(payload.get('entries', []))}"
        )
    return "; ".join(parts) if parts else None


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data" / "processed" / "yolo" / "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20,
                    help="早停等待轮数；设为 epochs 可关闭本轮训练内的早停")
    ap.add_argument("--imgsz", type=int, default=1024, help="与 patch 同尺度，避免 resize 失真")
    ap.add_argument("--batch", type=int, default=-1, help="-1=自动（8GB 显存约束）")
    ap.add_argument("--scale", type=float, default=0.5, help="多尺度增强 (±50%%)")
    ap.add_argument("--device", default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker 数；Windows 默认 0，避免路径/子进程写盘异常")
    ap.add_argument("--cache", default=None, choices=[None, "ram", "disk"],
                    help="缓存图片，ram 可减少重复读盘；默认不缓存")
    ap.add_argument("--resume", action="store_true",
                    help="从 --model 指定的 last.pt 继续训练")
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="类别加权指数（本仓库 ultralytics fork 内置）；"
                         "0.0=禁用，1.0=完全反频率，用于缓解类别不平衡导致的坍缩")
    ap.add_argument("--cls", type=float, default=None,
                    help="分类损失权重（ultralytics 原生超参，默认 0.5）；"
                         "单类别 Stage1 中即“滑坡/背景”置信度分支的学习强度")
    ap.add_argument("--cls-loss", choices=["bce", "vfl"], default="bce",
                    help="分类损失形式；默认 bce，vfl=Varifocal Loss（只替换分类分支）")
    ap.add_argument("--name", default="yolov8n_mars")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机种子；复验协议需要 seeds 0/1/2 分别跑")
    ap.add_argument("--manifest", default=None,
                    help="数据集 manifest 名或 json 路径，写入运行报告")
    ap.add_argument("--manifest-dir", default="data/manifest",
                    help="manifest 所在目录")
    ap.add_argument("--run-report-note", default=None,
                    help="写入运行报告的备注")
    ap.add_argument("--ignore-manifest", default=None,
                    help="训练期 ignore sidecar；只在内存中标记 train 框，不修改原始标签")
    ap.add_argument("--no-plots", action="store_true",
                    help="关闭训练标签/样本绘图（ignore 实验建议开启，避免绘图器看到内存 sentinel）")
    ap.add_argument("--no-run-report", dest="run_report", action="store_false",
                    help="训练后不生成运行报告")
    ap.set_defaults(run_report=True)
    return ap


def main():
    args = build_arg_parser().parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"[错误] 未找到 {data}，请先运行 make_yolo_dataset.py")
    if not (data.parent / "images" / "test").exists():
        print("[警告] 未见 test 集目录，确认 make_yolo_dataset.py 已生成独立 test 集")

    # 延迟 import，避免 torch 未就绪时脚本导入即崩
    from ultralytics import YOLO
    import torch

    if args.ignore_manifest:
        ignore_manifest = Path(args.ignore_manifest)
        if not ignore_manifest.exists():
            raise SystemExit(f"[错误] 未找到 ignore sidecar: {ignore_manifest}")
        from yolo_ignore import install_ignore_patch

        install_ignore_patch(ignore_manifest)
        print(f"[信息] 检测级 ignore 已启用（sidecar={ignore_manifest}，原始标签不改）")

    if args.cls_loss == "vfl":
        from vfl_loss import install_varifocal_loss
        import ultralytics.utils.loss as ultralytics_loss

        install_varifocal_loss(ultralytics_loss)
        print("[信息] Varifocal Loss 已启用：alpha=0.75, gamma=2.0；其余检测损失保持不变")

    _raw_torch_save = torch.save

    def _robust_torch_save(*save_args, **save_kwargs):
        """Windows 下内存缓冲区偶发被提前关闭；先序列化到独立缓冲区再复制。"""
        delay = 0.5
        target = save_args[1] if len(save_args) > 1 else save_kwargs.get("f")
        if hasattr(target, "write") and hasattr(target, "getvalue"):
            import io
            payload = save_args[0]
            rest = save_args[2:]
            for attempt in range(6):
                try:
                    tmp = io.BytesIO()
                    _raw_torch_save(payload, tmp, *rest, **save_kwargs)
                    data = tmp.getvalue()
                    target.seek(0)
                    target.truncate(0)
                    target.write(data)
                    target.seek(0)
                    return None
                except (RuntimeError, ValueError, OSError):
                    if attempt == 5:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
            return None
        for attempt in range(6):
            try:
                return _raw_torch_save(*save_args, **save_kwargs)
            except (RuntimeError, ValueError, OSError):
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

    torch.save = _robust_torch_save

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        scale=args.scale,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        resume=args.resume,
        project=str(PROJECT.resolve()),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        plots=not args.no_plots,
    )
    if args.cls_pw > 0.0:
        train_kwargs["cls_pw"] = args.cls_pw
        print(f"[信息] 类别加权已启用：cls_pw={args.cls_pw}")
    if args.cls is not None:
        train_kwargs["cls"] = args.cls
        print(f"[信息] 分类损失权重覆盖：cls={args.cls}")
    results = model.train(**train_kwargs)

    print("\n=== 验证（val 集）===")
    metrics = model.val(device=args.device, workers=args.workers,
                        project=str(PROJECT.resolve()), name=f"{args.name}_val",
                        exist_ok=True)
    print(metrics.results_dict)

    if args.run_report:
        try:
            from write_run_report import build_report, write_report

            trainer = getattr(model, "trainer", None)
            save_dir = Path(getattr(trainer, "save_dir", PROJECT / args.name))
            weight_files = [save_dir / "weights" / f"{kind}.pt" for kind in ("best", "last")]
            report = build_report(
                name=args.name,
                manifest=args.manifest,
                manifest_dir=args.manifest_dir,
                args_path=save_dir / "args.yaml",
                seed=args.seed,
                weights=[str(p) for p in weight_files if p.exists()],
                note=_build_run_note(args.run_report_note, args.ignore_manifest),
            )
            out_path = BASE / "runs" / f"{args.name}_run_report.json"
            write_report(report, out_path)
            print(f"[信息] 运行报告已写入 {out_path}")
            print(f"        report_sha256={report['report_sha256']}")
        except Exception as exc:
            print(f"[警告] 运行报告生成失败：{exc}")


if __name__ == "__main__":
    main()
