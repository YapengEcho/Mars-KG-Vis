"""Prepare CPU-only classifier development groups; never split or train a model."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'data/processed/manual_stage2_20261010_v1/manifest.json'
DEFAULT_OUT = ROOT / 'runs/research/manual_classification_protocol_20261014_v1'
CLASS_NAMES = ('SL', 'SP', 'FL', 'CL')
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SUSPECT_IDS = {'1770', '1771', '636', '637'}


def canonical_source_id(value):
    """Catalog IDs are positive integers, including legacy zero-padded strings."""
    text = str(value).strip()
    if not text.isdecimal() or int(text) < 1:
        raise ValueError(f'Invalid source_id: {value!r}')
    return str(int(text))


def counts(rows):
    counter = Counter(row['catalog_class'] for row in rows)
    return {name: counter[name] for name in CLASS_NAMES}


def source_rows(samples):
    """Deduplicate catalog IDs, merging all tile relationships conservatively."""
    appearances = defaultdict(list)
    for sample in samples:
        source_id = canonical_source_id(sample['source_id'])
        name = sample['catalog_class']
        if name not in CLASS_TO_ID:
            raise ValueError(f'Unknown catalog class for {source_id}: {name!r}')
        if sample.get('split') != 'train' or sample.get('used_by_v8_detector_training') is not True:
            raise ValueError(f'{source_id} is not an acknowledged detector-training source')
        if type(sample.get('class_reviewed')) is not bool:
            raise ValueError(f'{source_id} needs an explicit boolean class_reviewed')
        tiles = sample.get('source_tiles')
        if not isinstance(tiles, list) or not tiles or any(not isinstance(t, str) or not t.strip() for t in tiles):
            raise ValueError(f'{source_id} has no valid source_tiles')
        appearances[source_id].append(sample)
    rows = []
    for source_id in sorted(appearances, key=int):
        members = appearances[source_id]
        classes = {r['catalog_class'] for r in members}
        if len(classes) != 1:
            raise ValueError(f'Conflicting classes for source_id {source_id}: {sorted(classes)}')
        # Stable selection is metadata only. Other crop references remain auditable.
        selected = min(members, key=lambda r: (r.get('crop', ''), r.get('source_stem', '')))
        reviewed = all(r['class_reviewed'] for r in members)
        rows.append({
            'source_id': source_id,
            'catalog_class': selected['catalog_class'],
            'class_id': CLASS_TO_ID[selected['catalog_class']],
            'class_reviewed': reviewed,
            'source_tiles': sorted({t.strip() for r in members for t in r['source_tiles']}),
            'appearance_records': len(members),
            'crop': selected.get('crop'),
            'crop_sha256': selected.get('crop_sha256'),
            'crop_references': sorted({r['crop'] for r in members if r.get('crop')}),
            'source_stem': selected.get('source_stem'),
            'padding_fraction': selected.get('padding_fraction'),
            'longest_target_side_px': selected.get('longest_target_side_px'),
            'exploratory_eligible': source_id not in SUSPECT_IDS,
            'subtype_training_approved': reviewed and source_id not in SUSPECT_IDS,
            'hold_reason': ('suspected_neighbor_identity_requires_human_recheck'
                            if source_id in SUSPECT_IDS else None),
        })
    return rows


def spatial_components(rows):
    """Union tile nodes linked by one source; retain held nodes in the graph."""
    parent = {}

    def find(tile):
        parent.setdefault(tile, tile)
        if parent[tile] != tile:
            parent[tile] = find(parent[tile])
        return parent[tile]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for row in rows:
        first, *remaining = row['source_tiles']
        find(first)
        for tile in remaining:
            union(first, tile)
    components = defaultdict(list)
    for tile in sorted(parent):
        components[find(tile)].append(tile)
    groups = []
    for tiles in sorted(components.values(), key=lambda v: tuple(v)):
        tile_set = set(tiles)
        members = [r for r in rows if tile_set.intersection(r['source_tiles'])]
        # Each source must be wholly inside one connected component.
        if any(not set(r['source_tiles']) <= tile_set for r in members):
            raise AssertionError('A cross-tile source was split across groups')
        group_id = 'tile_component_' + hashlib.sha256('\n'.join(tiles).encode()).hexdigest()[:12]
        for row in members:
            row['spatial_group_id'] = group_id
        eligible = [r for r in members if r['exploratory_eligible']]
        groups.append({
            'spatial_group_id': group_id,
            'source_tiles': tiles,
            'source_ids': [r['source_id'] for r in members],
            'unique_targets': len(members),
            'catalog_class_counts': counts(members),
            'exploratory_eligible_targets': len(eligible),
            'eligible_catalog_class_counts': counts(eligible),
            'held_source_ids': [r['source_id'] for r in members if not r['exploratory_eligible']],
            'independently_reviewed_subtypes': sum(r['class_reviewed'] for r in members),
            'assigned_fold': None,
        })
    return groups


def make_protocol(manifest, manifest_path, manifest_sha256):
    rows = source_rows(manifest['samples'])
    groups = spatial_components(rows)
    eligible = [r for r in rows if r['exploratory_eligible']]
    class_groups = {
        name: sum(g['eligible_catalog_class_counts'][name] > 0 for g in groups)
        for name in CLASS_NAMES
    }
    return {
        'schema_version': 1,
        'status': 'development_groups_prepared_no_folds_or_models_created',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_manifest': str(Path(manifest_path).resolve()),
        'input_manifest_sha256': manifest_sha256,
        'input_appearance_records': len(manifest['samples']),
        'unique_targets': len(rows),
        'catalog_class_counts': counts(rows),
        'spatial_group_count': len(groups),
        'source_tile_node_count': len({t for r in rows for t in r['source_tiles']}),
        'cross_tile_targets': sum(len(r['source_tiles']) > 1 for r in rows),
        'exploratory_eligible_targets': len(eligible),
        'eligible_catalog_class_counts': counts(eligible),
        'held_targets': len(rows) - len(eligible),
        'approved_subtype_training_targets': sum(r['subtype_training_approved'] for r in rows),
        'eligible_groups_per_catalog_class': class_groups,
        'nominal_all_class_cv_fold_ceiling': min(class_groups.values()),
        'folds_assigned': False,
        'validation_ready': False,
        'test_ready': False,
        'class_mapping': {
            'classifier_class_to_id': CLASS_TO_ID,
            'classifier_class_names': list(CLASS_NAMES),
            'label_source': 'catalog_class in this manifest, followed by separate human subtype adjudication',
            'current_detector_class_to_id': {'LANDSLIDE': 0},
            'warning': 'Detector YOLO class 0 is LANDSLIDE, never evidence of subtype SL.',
        },
        'group_policy': {
            'method': 'connected components of source_tiles joined by shared source_id',
            'deduplication': 'one catalog source_id per classifier target; repeated appearances do not add independent targets',
            'holds_preserve_graph_connections': True,
            'scope': 'training-internal classifier development groups, not independent end-to-end evaluation',
            'fold_assignment': 'pending subtype review and geographic-overlap/buffer checks; no random image split',
            'limitation': 'Tile identifiers and manifest relationships do not prove the absence of spatial adjacency or crop-context overlap between components.',
        },
        'encoder_preset': {
            'name': 'independent_frozen_imagenet_resnet18',
            'status': 'planned_not_loaded_or_downloaded',
            'feature_endpoint': 'verified average-pooled representation; resolve by module name and shape at extraction time',
            'weights_receipt_required': 'record exact provider weight version and file SHA256 before extraction',
            'training_mode': 'eval; gradients disabled; no encoder fitting on classifier folds',
            'fold_preprocessing': 'fit scaler and any data-driven transforms on each fold training portion only',
            'heads_to_compare': ['logistic_regression', 'linear_svm'],
            'p2_layer_indices_allowed': False,
            'warning': 'Do not reuse legacy YOLO layers 15/18/21 for P2. A detector trained on these regions cannot supply independent pipeline evaluation features.',
        },
        'evaluation_plan': {
            'type_ground_truth': 'independently review SL/SP/FL/CL; FL/SP first, and resolve identity holds',
            'training_internal_diagnostics': 'group-held development metrics may be reported explicitly as catalog-label diagnostics, not acceptance scores',
            'independent_type_val_test': 'needs new reviewed targets in regions excluded from detector and classifier training',
            'metrics': ['per_class_precision_recall_f1', 'macro_f1', 'confusion_matrix', 'unique_source_count_per_class'],
            'paired_roi_comparison': 'compare human-box ROI and matched detector-box ROI for the same reviewed sources after detector training',
            'end_to_end_denominator': 'include detector misses; classifier accuracy on detected objects alone is insufficient',
            'uncertain_type_policy': 'retain a confirmed landslide as type_pending; low subtype confidence is not background evidence',
            'background_ground_truth': 'separate confirmed background regions; reject unknown, held, or merely misaligned positive candidates as negatives',
        },
        'resource_receipt': {
            'cpu_only': True, 'images_decoded': 0, 'gpu_model_loaded': False,
            'training_started': False, 'source_manifest_modified': False,
            'current_p2_runner_or_protocol_modified': False,
        },
        'groups': groups,
        'targets': rows,
    }


def prepare(manifest_path=DEFAULT_MANIFEST, out=DEFAULT_OUT):
    manifest_path, out = Path(manifest_path), Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior protocol; choose a new output directory')
    source_bytes = manifest_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    protocol = make_protocol(json.loads(source_bytes), manifest_path, source_sha)
    # Validate the entire plan before creating any output.
    held = [r for r in protocol['targets'] if not r['exploratory_eligible']]
    review = [r for r in protocol['targets'] if r['catalog_class'] in {'FL', 'SP'} and not r['class_reviewed']]
    out.mkdir(parents=True)

    def write_json(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    write_json('protocol.json', protocol)
    write_json('spatial_groups.json', {'status': protocol['status'], 'groups': protocol['groups']})
    write_json('pending_FL_SP_review.json', {'status': 'pending_human_subtype_review', 'targets': len(review), 'classes': counts(review), 'items': review})
    write_json('suspect_identity_holds.json', {
        'status': 'held_for_identity_recheck', 'requested_source_ids': sorted(SUSPECT_IDS, key=int),
        'targets': len(held), 'items': held,
        'note': 'Metadata-only holds; original reviewed boxes and crop manifest remain unchanged.',
    })
    lines = [
        '# 四分类准备协议', '',
        '本次只读取清单、准备分组和审核队列。没有读入全部图片，没有加载 GPU 模型，没有开始训练，没有改变 P2 或原始清单。', '',
        f"现有 {protocol['unique_targets']} 个独立目标，目录类别为 {protocol['catalog_class_counts']}；独立确认类型的目标为 {protocol['approved_subtype_training_targets']} 个。人工确认了框的位置，不等于确认了类型。", '',
        f"按 source_id 去重后，用同一目标出现的全部 source_tiles 建立连接关系，得到 {protocol['spatial_group_count']} 个训练内部空间组。跨图幅目标和其两侧图幅都绑定在同一组。共有 {protocol['cross_tile_targets']} 个多图幅目标。", '',
        '这不是新的验证集或测试集。本清单全部来自检测器训练区域，未分配任何折号。图幅标识连接只阻止已知关系的拆分，后续仍需检查相邻区域、裁片上下文重叠并设置空间缓冲。不能把训练内部组的成绩当成整个检测分类流程的独立验收成绩。', '',
        f"1770、1771、636、637 暂扣为疑似相邻编号或对应框问题，待人工复核；探索训练可用候选为 {protocol['exploratory_eligible_targets']} 个，类别为 {protocol['eligible_catalog_class_counts']}。此举不修改原框，也不认定它们不是滑坡。", '',
        f"FL/SP 优先类型审核队列有 {len(review)} 个；类别存在的空间组数为 {protocol['eligible_groups_per_catalog_class']}。这只是折数的必要条件，不保证类别数量足够、每折都能平衡或地域真正独立。", '',
        '## 类别映射', '',
        '分类映射固定为 SL=0、SP=1、FL=2、CL=3，读取 catalog_class 并叠加后续独立类型复核。当前检测器只输出 LANDSLIDE=0；检测标签中的 0 不能当成 SL。', '',
        '## 训练前需要补齐', '',
        '1. 独立确认类型，优先 FL/SP，解决四个编号暂扣问题。',
        '2. 先用独立冻结 ImageNet ResNet18 表示加 LR、线性 SVM 做基线；提特征前记录具体权重版本和 SHA256，核验模块名称与形状。',
        '3. 不沿用 P2 的固定层编号 15/18/21。标准化、特征筛选、类别平衡等拟合只看每折的训练部分。',
        '4. 分类验证/测试需要检测器和分类器都没有见过的新区域，补人工类型确认；当前全部为训练区，不能随机拿一部分称为独立测试。',
        '5. 检测完成后，同一批可靠目标比较人工框裁片和真实检测框裁片，报告逐类召回、macro-F1 和混淆矩阵。最终流程必须把漏检计入分母。类型分不清保留“类型待定”，不自动当背景删除。', '',
        '## 文件', '',
        '- protocol.json：输入哈希、类别映射、编码器预设、评估约束、全部独立目标和空间组。',
        '- spatial_groups.json：每组图幅、编号和四类数量。',
        '- pending_FL_SP_review.json：尚未独立确认的少数类。',
        '- suspect_identity_holds.json：四个疑似编号对应问题。',
        '- receipt.json：本协议文件哈希与原输入保持不变的核对。', '',
        f'原输入 SHA256：{source_sha}', '',
    ]
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError('Source manifest changed during preparation')
    receipt = {
        'input_manifest_sha256_before': source_sha,
        'input_manifest_sha256_after': source_sha,
        'source_unchanged': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
    }
    write_json('receipt.json', receipt)
    return {key: protocol[key] for key in (
        'status', 'unique_targets', 'catalog_class_counts', 'spatial_group_count',
        'source_tile_node_count', 'cross_tile_targets', 'exploratory_eligible_targets',
        'eligible_catalog_class_counts', 'held_targets', 'approved_subtype_training_targets',
        'eligible_groups_per_catalog_class', 'validation_ready', 'test_ready',
    )}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.out), ensure_ascii=False))
