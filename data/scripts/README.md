# 模型研究入口

这里保留数据构建、人工修框、检测、四分类与联合评估代码。它们与移动端 API 分开；提交这些脚本不代表识别服务已经接入 APP。

| 任务 | 入口 |
|---|---|
| 人工修正正样本、排除目标及保存背景框 | `manual_box_review.py`、同名 HTML |
| 审核评估图与额外候选 | `review_eval_labels.py`、`review_unmatched_candidates.py`、同名 HTML |
| 导出人工检测样本与核对空间划分 | `export_manual_detection_dataset.py`、`audit_yolo_splits.py` |
| SHP 外接框检测数据与基础训练 | `make_yolo_dataset.py`、`train_yolo.py` |
| 四分类空间分组与特征对照 | `prepare_manual_classification_protocol.py`、`run_fourclass_feature_comparison.py` |
| DINOv2 分类头与 LoRA 对照 | `run_classifier_lora_multiseed.py`、`run_classifier_head_rebalance_native40.py` |
| 分类编码器对照 | `run_classifier_encoder_arms.py`、`run_convnext_stage_repeat_20261023.py` |
| 滑坡／背景 LR、SVM 对照 | `run_binary_lr_svm_20261017.py`、`run_binary_lr_svm_search_20261017.py` |
| YOLO 训练配方与优化器对照 | `run_detector_recipe_factorial_20261020.py`、`run_detector_optimizer_20261022.py`、`run_detector_optimizer_repeat_20261024.py` |
| 检测框接四分类 | `run_latest_detector_classifier_bridge_20261024.py`、`run_optimizer_joint_bridge_20261025.py` |
| 固定候选预算融合与独立回算 | `tools/maintenance/` 中的 `run_optimizer_rank_ensemble_20261025.py` 及对应 `verify_*.py` |

## 运行条件

从仓库根目录运行。人工工具默认绑定本机地址；审核服务需要相应的私有清单、影像和审核包。先使用 `--help` 确认输入，不能把代码中的历史默认目录当作当前发布数据。

```bash
python data/scripts/manual_box_review.py --help
python data/scripts/make_yolo_dataset.py --help
python data/scripts/train_yolo.py --help
```

基础数据工具依赖 NumPy、Pillow、pyshp、tifffile、SciPy、PyYAML。模型研究另需 PyTorch、Torchvision、Ultralytics、scikit-learn、joblib、timm、Transformers、psutil；CUDA 安装应匹配实际硬件。CPU 测试所需的轻量依赖见 `requirements-test.txt`。

日期结尾的脚本是历史受控实验入口，包含固定目录、协议哈希和 Windows 后台启动方式。运行前必须提供协议引用的私有数据、权重、模型实现及运行环境；这些文件缺失时不能完整复现实验。历史训练还依赖当时使用的 Ultralytics 实现，通用训练入口中的部分参数并非任意官方版本都支持。仅安装最新版依赖不能保证数值一致。

本次提交按所选入口的本地 Python 依赖及显式脚本引用整理，包含必要的历史依赖；没有迁入全部个人研究目录。原有 SHP、人工记录、专家类型与空间划分均不在公开仓库中。

## 检查

安装轻量依赖后，以下测试不读取私有数据、不启动训练或下载：

```bash
python -m pip install -r data/scripts/requirements-test.txt
python -m pytest tests --ignore=tests/test_manual_box_experiment.py -q
python -m pytest data/scripts/test_stage2_features.py -q
node tests/manual_box_review_frontend.cjs
```

`tests/test_manual_box_experiment.py` 还需安装兼容的 PyTorch 和 Ultralytics，使用合成数据及替身对象检查训练配置，不运行完整训练。`data/scripts/test_eval_cascade.py` 的匹配测试也会导入 PyTorch。上述两个文件在完整研究环境中另行运行，轻量 CI 不安装模型依赖。真实模型实验的回算需另行提供对应私有产物。

研究范围与最新结果见 [研究说明](../../docs/research/README.md)。
