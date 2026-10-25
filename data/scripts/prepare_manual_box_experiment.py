"""Freeze the v4 three-arm comparison; this command never trains a model."""
import argparse
from collections import Counter
import csv
import json
import platform
from pathlib import Path
import shutil

import torch
import ultralytics
import yaml

from export_manual_detection_dataset import BASE, OLD, read_json, write_json, verify_labels
from build_center_candidate_manifest import sha256_file

DATASET = BASE / 'data/processed/manual_yolo_20261008_v5'
EXPERIMENT = BASE / 'runs/experiments/manual_box_v5_20261008'


def environment():
    package=Path(ultralytics.__file__).parent
    paths=['engine/trainer.py','models/yolo/detect/train.py','data/dataset.py',
           'data/augment.py','utils/loss.py','cfg/default.yaml','nn/modules/head.py']
    return dict(python=platform.python_version(),torch=torch.__version__,
                ultralytics=ultralytics.__version__,cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                bf16_supported=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                package_code={str(package/p):sha256_file(package/p) for p in paths})


def prepare(out=EXPERIMENT,dataset=DATASET):
    DATASET=dataset.resolve()
    out=out.resolve()
    if out.exists():raise FileExistsError('Use a new experiment directory; frozen specifications are not overwritten.')
    assert read_json(DATASET/'build_state.json')['status']=='complete_data_export_no_model_training'
    out.mkdir(parents=True);(out/'datasets').mkdir();(out/'inputs').mkdir();(out/'code_snapshot').mkdir()
    env=environment();write_json(out/'environment.json',env)
    if not env['cuda_available'] or not env['bf16_supported']:raise RuntimeError('Protocol requires this CUDA/BF16 environment')
    weight=out/'inputs/yolov8n_initial.pt';shutil.copy2(BASE/'yolov8n.pt',weight)
    plan=read_json(DATASET/'export_plan.json');old=read_json(Path(plan['inputs']['old_manifest']['snapshot']))
    report=read_json(DATASET/'build_report.json')
    historical_hashes={r['path']:r['sha256'] for r in old['files']}
    new_hashes={s['stem']+'.png':s['image_sha256'] for s in plan['samples']}
    arms={};checks={};eval_identity={};training_lists={}
    for arm,root in [('manual',DATASET/'manual'),('shp_control',DATASET/'shp_control'),('historical',OLD)]:
        source_list=root/('train_stage1_rebalanced.txt' if arm=='historical' else 'train_equal_exposure.txt')
        lines=[str((root/'images/train'/Path(s).name).resolve()).replace('\\','/') for s in source_list.read_text(encoding='utf-8').splitlines()]
        assert len(lines)==4868
        expected_unique=2880 if arm=='historical' else len(plan['samples'])
        assert len(set(lines))==expected_unique
        assert set(Path(x).name for x in lines)=={p.name for p in (root/'images/train').glob('*.png')}
        training_lists[arm]=[Path(x).name for x in lines]
        trainlist=out/'datasets'/f'{arm}_train.txt';trainlist.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        data=out/'datasets'/f'{arm}.yaml'
        data.write_text(yaml.safe_dump(dict(path=root.as_posix(),train=trainlist.as_posix(),
                       val='images/val',test='images/test',names={0:'LANDSLIDE'}),sort_keys=False),encoding='utf-8')
        records=[]
        for split in ['train','val','test']:
            images=sorted((root/'images'/split).glob('*.png'))
            assert len(images)==(expected_unique if split=='train' else 615 if split=='val' else 497)
            for image in images:
                label=root/'labels'/split/(image.stem+'.txt');verify_labels(label)
                for path in [image,label]:
                    digest=sha256_file(path);rel=path.relative_to(root).as_posix()
                    if arm=='historical' or split!='train':assert digest==historical_hashes[rel],str(path)
                    elif path==image:assert digest==new_hashes[image.name],str(path)
                    records.append(dict(path=str(path.resolve()),sha256=digest))
                if split!='train':
                    identity=(sha256_file(image),sha256_file(label))
                    key=f'{split}/{image.name}'
                    if key in eval_identity:assert eval_identity[key]==identity
                    else:eval_identity[key]=identity
        checks[arm]=records
        arms[arm]=dict(data=str(data),data_sha256=sha256_file(data),train_list=str(trainlist),
            train_list_sha256=sha256_file(trainlist),train_entries=4868,unique_images=expected_unique,
            positive_images=1334 if arm=='historical' else report['train_positive_images'],negative_images=1546 if arm=='historical' else report['train_negative_images'])
        print('FROZEN_ARM',arm,expected_unique,flush=True)
    assert training_lists['manual']==training_lists['shp_control']
    for name in training_lists['manual']:
        assert sha256_file(DATASET/'manual/images/train'/name)==sha256_file(DATASET/'shp_control/images/train'/name)
    training=dict(epochs=150,patience=0,imgsz=1024,batch=4,nbs=64,workers=0,cache=False,
        device=0,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,
        warmup_epochs=3.0,warmup_momentum=.8,warmup_bias_lr=.1,cos_lr=False,
        amp='bf16',deterministic=True,resume=False,pretrained=True,single_cls=False,
        rect=False,multi_scale=0.0,compile=False,channels_last=False,
        box=7.5,cls=.5,cls_pw=0.0,dfl=1.5,cls_remap=True,
        hsv_h=.015,hsv_s=.7,hsv_v=.4,degrees=0.0,translate=.1,scale=.5,
        shear=0.0,perspective=0.0,flipud=0.0,fliplr=.5,bgr=0.0,
        mosaic=1.0,mixup=0.0,copy_paste=0.0,close_mosaic=10,
        val=True,save=True,save_period=-1,plots=False,exist_ok=False)
    specification=dict(schema_version=1,status='prepared_formal_training_not_started',
        dataset_root=str(DATASET),dataset_plan_sha256=sha256_file(DATASET/'export_plan.json'),
        weights=str(weight),weights_sha256=sha256_file(weight),environment=env,
        arms=arms,seeds=[0,1,2],training=training,
        eval=dict(conf_pool=.001,nms_iou=.7,match_iou=.5,max_det=300,
            threshold_selection='validation only: maximize recall within 2455 total candidates; ties: precision, fewer candidates, higher threshold',
            candidate_budget=2455,checkpoint='best.pt selected on validation fitness',
            threshold_grid=[.001,.005,.01,.02,.03,.05,.075,.1,.15,.25,.4,.5,.7,.9,1.0]),
        comparisons=dict(label_quality=['manual','shp_control'],whole_dataset=['manual','historical']),
        expected_batches_per_epoch=1217,formal_training_runs=9,
        limitations=['Historical evaluation retains old AABB annotations.',
                    'Historical comparison changes image crops, counts, backgrounds and sampling composition.',
                    'Image-list exposure is equal; mosaic adds supplementary reads. Actual optimizer updates are audited.'])
    write_json(out/'input_checksums.json',checks)
    specification['input_checksums_sha256']=sha256_file(out/'input_checksums.json')
    write_json(out/'experiment.json',specification)
    (out/'experiment.sha256').write_text(sha256_file(out/'experiment.json')+'\n',encoding='utf-8')
    for name in ['prepare_manual_box_experiment.py','run_manual_box_experiment.py','evaluate_manual_box_experiment.py']:
        src=Path(__file__).with_name(name)
        if src.exists():shutil.copy2(src,out/'code_snapshot'/name)
    write_json(out/'code_fingerprints.json',{str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*manual_box_experiment.py')})
    rows=[dict(arm=a,seed=s,train_unique_images=arms[a]['unique_images'],val_map50='',val_map50_95='',
               locked_threshold='',test_precision='',test_recall='',test_fp_per_image='',actual_optimizer_steps='')
          for s in specification['seeds'] for a in arms]
    with (out/'results_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    commands='\n'.join(f"& .venv\\Scripts\\python.exe data/scripts/run_manual_box_experiment.py train --experiment '{out}' --arm {a} --seed {s}" for s in specification['seeds'] for a in arms)
    (out/'run_all.ps1').write_text("Set-Location -LiteralPath 'D:\\Mars-KG-Vis'\n$ErrorActionPreference = 'Stop'\n"+'\n'.join(c+"\nif ($LASTEXITCODE -ne 0) { throw 'Experiment failed; later arms were not started.' }" for c in commands.splitlines())+'\n',encoding='utf-8')
    description='''# 人工框 v4 对照实验

准备完成后，正式训练仍需单独启动。三臂均从同一份 YOLOv8n 初始权重重新训练。

| 组别 | 唯一训练图像 | 正裁片 | 负裁片 | 主要用途 |
|---|---:|---:|---:|---|
| manual | 2309 | 954 | 1355 | 人工修订框 |
| shp_control | 2309 | 954 | 1355 | 同图同负样本的原SHP框 |
| historical | 2880 | 1334 | 1546 | 旧完整数据方案 |

manual与shp_control图像、顺序、负样本、目标ID和划分一致；比较框修订的影响。
manual与historical比较整套数据更新的效果，不能单独归因于框修订。

三臂：150轮、1024输入、固定batch=4、SGD、BF16、nbs=64、关闭早停，种子0/1/2，共9次正式训练。
每轮4868条训练输入、1217个batch；记录实际优化器更新数、初始化模型哈希及首个增强后图像batch哈希。
同一GPU串行运行。Windows workers=0，不使用自动batch或自动优化器。

验证615张、测试497张，所有臂图像和标签相同，已知训练滑坡与评估划分隔离。
验证集选择best.pt及工作阈值：候选总数不超过2455时最大化目标召回率，随后比较精度、候选数及阈值。
锁定权重和阈值后评估测试集一次；不得用测试结果重新选阈值或模型。
记录mAP50、mAP50-95、精度、召回率、FP/image、空背景误报比例、目标尺度分组召回。
三种子报告均值、样本标准差及每个种子的原始结果；不从少量图幅宣称统计显著。
历史验证/测试仍使用旧标注，未来需要另行审核独立评估图幅。

命令在项目根目录执行：

```powershell
# 仅8条训练输入的运行检查，结果不计入正式对照
.venv\\Scripts\\python.exe data/scripts/run_manual_box_experiment.py smoke
# 正式单臂示例
.venv\\Scripts\\python.exe data/scripts/run_manual_box_experiment.py train --arm manual --seed 0
# 九次正式训练顺序执行
& runs/experiments/manual_box_v4_20261008/run_all.ps1
# 训练后的验证集阈值选择和锁定
.venv\\Scripts\\python.exe data/scripts/evaluate_manual_box_experiment.py select --arm manual --seed 0
# 只有锁定后才允许测试集评估
.venv\\Scripts\\python.exe data/scripts/evaluate_manual_box_experiment.py test --arm manual --seed 0
```

experiment.json及其SHA固定协议；input_checksums.json固定数据文件；environment.json包含当前库源码哈希。
后续补审得到的新样本另建版本，不能加入正在运行的v4实验。
'''
    description=description.replace('v4',DATASET.name.rsplit('_',1)[-1]).replace('2309',str(len(plan['samples']))).replace('954',str(report['train_positive_images'])).replace('1355',str(report['train_negative_images']))
    description=description.replace('runs/experiments/manual_box_'+DATASET.name.rsplit('_',1)[-1]+'_20261008/run_all.ps1',(out/'run_all.ps1').as_posix())
    for command in ['run_manual_box_experiment.py smoke','run_manual_box_experiment.py train','evaluate_manual_box_experiment.py select','evaluate_manual_box_experiment.py test']:
        description=description.replace(command,command+f' --experiment "{out}"')
    (out/'README.md').write_text(description,encoding='utf-8')
    print('EXPERIMENT_PREPARED',out,flush=True)
    return specification


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=EXPERIMENT);ap.add_argument('--dataset',type=Path,default=DATASET);a=ap.parse_args();prepare(a.out,a.dataset)
