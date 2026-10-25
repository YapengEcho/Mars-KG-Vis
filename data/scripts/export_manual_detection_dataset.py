"""Freeze human annotations, recrop native CTX pixels, export paired YOLO datasets.

Only confirmed complete manual positives and selected human background rectangles
enter training. All known neighboring landslides must be fully annotated. Frozen
historical val/test are reused byte for byte; SHP and live annotations are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import shapefile
import tifffile

from build_center_candidate_manifest import (CLASSES, TYPE_ALIASES, contains,
    load_overrides, sha256_file, validate_box, yolo_box)
from check_tile_connectivity import tiles_of_bbox
from continue_ctx_review_download import protected_tiles
from make_centered_review_pack import (BASE, SHP, _intersects, _record_for_tile,
    _tile_bounds, discover_tifs, map_source_records, read_geotransform)
from manual_box_review import has_complete_geometry

PACK = BASE / 'data/processed/centered_review_v2_train_20261006'
OLD = BASE / 'data/processed/yolo_tile_flval_round3_multilabel_stage1'
OLD_MANIFEST = BASE / 'data/manifest/yolo_tile_flval_round3_multilabel_stage1.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def stretch_uint8(crop):
    """Exact historical percentiles via chunked histogram and a 256-entry LUT.

    Avoid gigabytes of float64 intermediate arrays for large native crop windows.
    Linear percentile interpolation follows numpy.percentile on nonzero pixels.
    """
    if crop.dtype != np.uint8 or crop.ndim != 2:
        raise ValueError('Brightness stretch expects uint8 grayscale')
    counts=np.zeros(256,dtype=np.int64)
    for start in range(0,crop.shape[0],64):
        counts+=np.bincount(crop[start:start+64].ravel(),minlength=256)
    counts[0]=0
    n=int(counts.sum())
    cumulative=np.cumsum(counts)
    def quantile(p):
        index=(n-1)*p
        lower,upper=math.floor(index),math.ceil(index)
        a=int(np.searchsorted(cumulative,lower,side='right'))
        b=int(np.searchsorted(cumulative,upper,side='right'))
        return a+(b-a)*(index-lower)
    lo,hi=(quantile(.02),quantile(.98)) if n else (0,255)
    if hi<=lo:
        hi=lo+1
    lut=np.clip((np.arange(256,dtype=np.float64)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
    lut[0]=0
    return lut[crop]


def snapshot_inputs(pack, out):
    """Retry coherent live reads, then pin immutable build inputs before planning."""
    paths = {'queue': pack / 'review_queue.csv',
             'overrides': pack / 'manual_review/manual_boxes.reviewed.csv',
             'exclusions': pack / 'manual_review/review_exclusions.json',
             'backgrounds': pack / 'manual_review/manual_background_boxes.json',
             'background_groups': pack / 'manual_review/background_batches_0001_0018.json',
             'last_batch': pack / 'manual_review/batch_0019.completed.json',
             'geometry_holds': pack / 'manual_review/geometry_holds.json',
             'old_manifest': OLD_MANIFEST, 'shp': SHP,
             'shx': SHP.with_suffix('.shx'), 'dbf': SHP.with_suffix('.dbf'),
             'final_eval': BASE / 'data/raw/final_eval_tiles_frozen.json'}
    for name in ('tile_split_plan_flval.json', 'tile_split_plan_flval_v2.json',
                 'tile_split_plan_flval_train_expansion.json'):
        paths[name] = BASE / 'data/raw' / name
    for tile_dir in sorted((pack / 'native_context').iterdir()):
        if (tile_dir / 'manifest.json').exists():
            paths['cache_' + tile_dir.name] = tile_dir / 'manifest.json'
    for _ in range(3):
        data = {k: p.read_bytes() for k, p in paths.items()}
        if all(p.read_bytes() == data[k] for k, p in paths.items()):
            break
    else:
        raise RuntimeError('Annotations changed during snapshot; retry when saves stop.')
    snap = out / 'inputs'
    snap.mkdir(parents=True)
    index = {}
    for key, path in paths.items():
        target = snap / (key.replace('.json', '') + path.suffix)
        target.write_bytes(data[key])
        index[key] = {'source': str(path), 'snapshot': str(target),
                      'sha256': hashlib.sha256(data[key]).hexdigest()}
    write_json(snap / 'index.json', index)
    return index


def inventory(index, selected_negative_batches=None):
    p = lambda k: Path(index[k]['snapshot'])
    with p('queue').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    queue = {(r['tile'], int(r['source_id'])): r for r in rows}
    if len(queue) != len(rows):
        raise ValueError('Duplicate queue keys')
    manual = load_overrides(p('overrides'), set(queue))
    exclusions = read_json(p('exclusions'))
    held_geometry = read_json(p('geometry_holds'))
    groups = read_json(p('background_groups'))['groups']
    batch_members = {}
    for g in groups:
        for r in g['items']:
            batch_members.setdefault((r['tile'], int(r['source_id'])), set()).add(g['batch'])
    for r in read_json(p('last_batch'))['items']:
        batch_members.setdefault((r['tile'], int(r['source_id'])), set()).add(19)
    selected = set(selected_negative_batches if selected_negative_batches is not None else list(range(1,16))+[19])
    backgrounds, not_selected = [], []
    for bid, r in read_json(p('backgrounds')).items():
        if bid != r['background_id']:
            raise ValueError('Background identity mismatch')
        r = dict(r, batches=sorted(batch_members.get((r['tile'], int(r['source_review_id'])), set())))
        (backgrounds if selected.intersection(r['batches']) else not_selected).append(r)
    protected = set(read_json(p('final_eval'))['tiles'])
    for key in index:
        if key.startswith('tile_split_plan_'):
            protected.update(t for t, s in read_json(p(key)).items() if s in {'val', 'test'})
    if protected != protected_tiles():
        raise RuntimeError('Split plan changed during build')
    return queue, manual, exclusions, held_geometry, backgrounds, not_selected, protected


def offset_window(box, available, key, margin=1.5):
    """Deterministic independent x/y context offsets; never clip the target box."""
    if not contains(available, box):
        return None
    digest = hashlib.sha256(key.encode()).digest()
    side = math.ceil(max(1024, (box[2]-box[0])*margin, (box[3]-box[1])*margin))
    w = min(side, available[2]-available[0])
    h = min(side, available[3]-available[1])
    if w < box[2]-box[0] or h < box[3]-box[1]:
        return None
    fx, fy = [0.1 + 0.8*v/255 for v in digest[:2]]
    x = math.floor(box[0] - (w-(box[2]-box[0]))*fx)
    y = math.floor(box[1] - (h-(box[3]-box[1]))*fy)
    x = max(available[0], min(x, available[2]-w))
    y = max(available[1], min(y, available[3]-h))
    result = [x, y, x+w, y+h]
    return result if contains(result, box) else None


def positive_window(primary_id, records, available, key, margin=1.5):
    """Close over intersecting approved neighbors; hold any unknown or excluded area."""
    target = records[primary_id]
    window = offset_window(target['support'], available, key, margin)
    if window is None:
        return None, [], 'native_context_does_not_contain_primary'
    for _ in range(len(records)+1):
        hits = [r for r in records.values() if _intersects(tuple(window), tuple(r['support']))]
        blocked = [r for r in hits if r['hold_reason']]
        if blocked:
            return None, [], 'neighbor_hold:' + ','.join(
                f"{r['source_id']}={r['hold_reason']}" for r in blocked)
        expanded = union([window] + [r['support'] for r in hits])
        expanded = [math.floor(expanded[0]), math.floor(expanded[1]),
                    math.ceil(expanded[2]), math.ceil(expanded[3])]
        if not contains(available, expanded):
            return None, [], 'neighbor_requires_unavailable_native_context'
        if expanded == window:
            return window, sorted(hits, key=lambda r: r['source_id']), None
        window = expanded
    raise RuntimeError('Neighbor closure failed to converge')


def background_reason(record, bounds, records):
    if record.get('confirmed') is not True or record.get('judgement') != 'certain_background':
        return 'background_not_human_confirmed'
    if not str(record.get('reviewer', '')).strip() or not str(record.get('reason', '')).strip():
        return 'background_missing_review_provenance'
    box = validate_box(record['aabb_px'])
    if any(v != int(v) for v in box):
        return 'background_not_whole_pixel_rectangle'
    if not contains(bounds, box) or not contains(record['view_window_px'], box):
        return 'background_outside_reviewed_pixels'
    collisions = [r['source_id'] for r in records.values()
                  if _intersects(tuple(box), tuple(r['support']))]
    return 'overlaps_known_or_manual_landslide:' + ','.join(map(str, collisions)) if collisions else None


def make_plan(pack, out, index, selected_negative_batches=None):
    queue, manual, exclusions, geometry_holds, backgrounds, not_selected, protected = inventory(index, selected_negative_batches)
    p = lambda k: Path(index[k]['snapshot'])
    with shapefile.Reader(shp=str(p('shp')), shx=str(p('shx')), dbf=str(p('dbf'))) as reader:
        sources = map_source_records(reader.shapeRecords())
    tifs = discover_tifs(BASE / 'data/raw/ctx_tiles')
    protected_sources = {sid for sid, r in sources.items() if tiles_of_bbox(*r.shape.bbox) & protected}
    # Unconfirmed sidecar rectangles still protect negatives and crop completeness.
    with p('overrides').open(encoding='utf-8-sig', newline='') as handle:
        all_boxes = {(r['tile'], int(r['source_id'])):
                     validate_box([r[f] for f in ('x0','y0','x1','y1')]) for r in csv.DictReader(handle)}
    samples, holds, tile_sources = [], [], {}
    seen_windows = set()
    tiles = sorted({t for t, _ in manual} | {r['tile'] for r in backgrounds})
    for tile in tiles:
        if tile in protected:
            raise ValueError('Protected review tile: ' + tile)
        cache = None
        if tile in tifs:
            gt, width, height = read_geotransform(tifs[tile])
            pixel_source = {'kind':'full_tile', 'path':str(tifs[tile]),
                            'bytes':tifs[tile].stat().st_size, 'mtime_ns':tifs[tile].stat().st_mtime_ns}
        else:
            cache = read_json(p('cache_' + tile))
            gt, width, height = cache['geotransform'], cache['tile_width'], cache['tile_height']
            pixel_source = {'kind':'cached_context', 'manifest':index['cache_'+tile]}
        bounds = [0, 0, width, height]
        records = {}
        for sid, source in sources.items():
            r = _record_for_tile(sid, source, gt, _tile_bounds(gt, width, height))
            if r is None:
                continue
            k = (tile, sid)
            override = manual.get(k)
            exclusion = exclusions.get(f'{tile}:{sid}')
            pieces = [r['aabb_px']]
            if k in all_boxes:
                pieces.append(all_boxes[k])
            if exclusion and exclusion.get('exclusion_box_px'):
                pieces.append(validate_box(exclusion['exclusion_box_px']))
            if exclusion:
                if exclusion.get('negative_eligible') is not False or exclusion.get('training_use') != 'exclude_positive':
                    raise ValueError('Exclusion policy changed')
                reason = 'human_exclusion_not_background'
            elif sid in protected_sources:
                reason = 'source_touches_protected_tile'
            elif k not in queue or not has_complete_geometry(queue[k]) or f'{tile}:{sid}' in geometry_holds:
                reason = 'unreviewed_or_incomplete_geometry'
            elif override is None:
                reason = 'missing_confirmed_manual_box'
            elif override['label_scope'] != 'full_landslide':
                reason = 'visible_core_requires_separate_task'
            else:
                reason = None
            r.update(support=union(pieces), manual_box=override['aabb_px'] if override else None,
                     hold_reason=reason, class_name=TYPE_ALIASES.get(r['raw_type'],r['raw_type']))
            if r['class_name'] not in CLASSES:
                raise ValueError('Unknown SHP class')
            if k in queue and any(abs(a-b)>0.01 for a,b in zip(r['aabb_px'], json.loads(queue[k]['polygon_bbox_px']))):
                raise ValueError('Queue geometry changed: ' + str(k))
            records[sid] = r
        tile_sources[tile] = dict(pixel_source, tile_size_px=[width,height], geotransform=list(gt))

        def available_for(box, review_id):
            if not cache:
                return bounds, pixel_source
            choices = [e for e in cache['entries'] if contains(e['window_origin'],box)]
            choices.sort(key=lambda e: (int(e['source_id']) != review_id,
                                       e['shape'][0]*e['shape'][1],e['source_id']))
            if not choices:
                return None, None
            e = choices[0]
            if e['stride'] != 1:
                raise ValueError('Downsampled native context is forbidden')
            return e['window_origin'], dict(kind='cached_context', **e)

        for _, sid in sorted(k for k in manual if k[0] == tile):
            r = records.get(sid)
            if r is None:
                raise ValueError('Manual source absent from SHP: ' + str((tile,sid)))
            reason = r['hold_reason']
            available, src = available_for(r['support'],sid)
            labels, window = [], None
            if reason is None:
                if available is None:
                    reason = 'native_context_does_not_contain_primary'
                else:
                    for margin in (1.5,1.25,1.1):
                        window, labels, reason = positive_window(sid,records,available,f'{tile}:{sid}',margin)
                        if reason is None:
                            break
            if reason:
                holds.append(dict(kind='positive',tile=tile,source_id=sid,reason=reason))
                continue
            identity = (tile,tuple(window))
            if identity in seen_windows:
                holds.append(dict(kind='positive',tile=tile,source_id=sid,reason='duplicate_crop_window'))
                continue
            seen_windows.add(identity)
            items = [dict(source_id=a['source_id'], class_name=a['class_name'],
                          manual_aabb_px=a['manual_box'], shp_aabb_px=a['aabb_px']) for a in labels]
            samples.append(dict(kind='positive', tile=tile,source_id=sid,window_px=window,
                pixel_source=src,labels=items,stem=f'{tile}_manual_{sid:05d}'))
        for b in sorted((b for b in backgrounds if b['tile']==tile),key=lambda b:b['background_id']):
            reason = background_reason(b,bounds,records)
            available, src = available_for(b['aabb_px'],int(b['source_review_id']))
            if reason is None and available is None:
                reason = 'background_native_pixels_unavailable'
            identity = (tile,tuple(b['aabb_px']))
            if reason is None and identity in seen_windows:
                reason = 'duplicate_crop_window'
            if reason:
                holds.append(dict(kind='negative',tile=tile,background_id=b['background_id'],reason=reason))
                continue
            seen_windows.add(identity)
            samples.append(dict(kind='negative',tile=tile,background_id=b['background_id'],
                source_review_id=b['source_review_id'],batches=b['batches'],window_px=b['aabb_px'],
                pixel_source=src,labels=[],stem=tile+'_'+b['background_id']))
    payload = dict(schema_version=1,status='planned_not_exported',generated_utc=datetime.now(timezone.utc).isoformat(),
        input_positive_boxes=len(manual),input_selected_background_boxes=len(backgrounds),
        input_exclusions=len(exclusions),unselected_background_count=len(not_selected),
        selected_negative_batches=list(selected_negative_batches) if selected_negative_batches is not None else list(range(1,16))+[19],protected_tiles=sorted(protected),
        protected_sources=sorted(protected_sources),tile_sources=tile_sources,
        samples=samples,held=holds,inputs=index,
        image_processing='native uint8, 2-98 percentile stretch shared with historical pipeline; aspect preserved; longest edge 1024 LANCZOS',
        crop_policy='offset context with full SHP/manual union and complete approved neighbor closure; margins 1.5,1.25,1.1; no ignore-loss',
        eval_policy='historical Stage1 val/test byte-identical in both arms; all reviewed data is train-only')
    write_json(out/'export_plan.json',payload)
    print(json.dumps(dict(planned_positive=sum(s['kind']=='positive' for s in samples),
        planned_negative=sum(s['kind']=='negative' for s in samples),held_reasons=dict(Counter(h['reason'].split(':')[0] for h in holds))),ensure_ascii=False),flush=True)
    return payload


class Pixels:
    def __init__(self):
        self.tile = None
        self.array = None
        self.origin = None
        self.checked = set()

    def crop(self,sample):
        src = sample['pixel_source']
        path = Path(src['path'])
        if self.tile != str(path):
            self.array = None
            if src['kind']=='full_tile':
                if path.stat().st_size!=src['bytes'] or path.stat().st_mtime_ns!=src['mtime_ns']:
                    raise ValueError('Native TIFF changed: '+str(path))
                self.array = tifffile.memmap(path)
                self.origin = [0,0,self.array.shape[1],self.array.shape[0]]
            else:
                if str(path) not in self.checked:
                    if sha256_file(path)!=src['sha256']:
                        raise ValueError('Native cache checksum mismatch')
                    self.checked.add(str(path))
                self.array = tifffile.imread(path)
                if list(self.array.shape)!=src['shape']:
                    raise ValueError('Native cache dimensions changed')
                self.origin = src['window_origin']
            self.tile = str(path)
        if self.array.dtype!=np.uint8 or not contains(self.origin,sample['window_px']):
            raise ValueError('Native pixels missing or wrong dtype')
        x0,y0,x1,y1 = sample['window_px']
        x,y = self.origin[:2]
        # Bounds must be exact whole pixels for recropping and normalized labels.
        if any(v != int(v) for v in (x0,y0,x1,y1)):
            raise ValueError('Fractional crop window')
        return np.asarray(self.array[int(y0-y):int(y1-y),int(x0-x):int(x1-x)]).copy()


def link_or_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256_file(source)!=sha256_file(target):
            raise ValueError('Existing export file changed: '+str(target))
        return
    try:
        os.link(source,target)
    except OSError:
        shutil.copy2(source,target)


def labels_text(sample,field,multiclass=False):
    w=sample['window_px'][2]-sample['window_px'][0]
    h=sample['window_px'][3]-sample['window_px'][1]
    lines=[]
    for label in sample['labels']:
        box=label[field]
        if not contains(sample['window_px'],box):
            raise ValueError('Clipped label forbidden')
        xywh=yolo_box(box,sample['window_px'],[w,h])
        cls=CLASSES[label['class_name']] if multiclass else 0
        lines.append(str(cls)+' '+' '.join(f'{v:.8f}' for v in xywh))
    return '\n'.join(lines)+('\n' if lines else '')


def verify_labels(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        fields=line.split()
        if len(fields)!=5 or fields[0]!='0':
            raise ValueError('Invalid Stage1 label: '+str(path))
        x,y,w,h=map(float,fields[1:])
        # Historical labels use six decimals: independent cx/width rounding can
        # put a boundary 0.00000075 outside. Keep their bytes; reject real excess.
        if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
            raise ValueError('Out-of-frame YOLO box: '+str(path))


def export(plan,out,old,reuse_verified_samples=False):
    roots={arm:out/arm for arm in ('manual','shp_control')}
    for root in roots.values():
        for split in ('train','val','test'):
            (root/'images'/split).mkdir(parents=True,exist_ok=True)
            (root/'labels'/split).mkdir(parents=True,exist_ok=True)
    frozen=read_json(Path(plan['inputs']['old_manifest']['snapshot']))
    expected={r['path']:r['sha256'] for r in frozen['files']}
    eval_records=[]
    image_hash_splits={}
    for split in ('val','test'):
        for image in sorted((old/'images'/split).glob('*.png')):
            label=old/'labels'/split/(image.stem+'.txt')
            hashes={}
            for src in (image,label):
                rel=src.relative_to(old).as_posix()
                hashes[rel]=sha256_file(src)
                if hashes[rel]!=expected.get(rel):
                    raise ValueError('Historical eval differs from frozen manifest: '+str(src))
            for root in roots.values():
                link_or_copy(image,root/'images'/split/image.name)
                target_label=root/'labels'/split/label.name
                if target_label.exists() and target_label.read_bytes()!=label.read_bytes():
                    raise ValueError('Existing eval label changed')
                if not target_label.exists():
                    shutil.copy2(label,target_label)
            verify_labels(label)
            sha=hashes[image.relative_to(old).as_posix()]
            image_hash_splits.setdefault(sha,set()).add(split)
            eval_records.append(dict(split=split,image=image.name,hashes=hashes,
                label_count=len(label.read_text(encoding='utf-8').splitlines())))
    if any(len(s)>1 for s in image_hash_splits.values()):
        raise ValueError('Historical val/test image hash overlap')
    actual=Counter(r['split'] for r in eval_records)
    for split in ('val','test'):
        if actual[split]!=frozen['summary']['splits'][split]['images']:
            raise ValueError('Incomplete frozen evaluation split')
    pixels=Pixels()
    for i,s in enumerate(plan['samples']):
        dst=roots['manual']/'images/train'/(s['stem']+'.png')
        if reuse_verified_samples and s.get('image_sha256'):
            if not dst.exists() or sha256_file(dst)!=s['image_sha256']:
                raise ValueError('Previously verified training image changed: '+str(dst))
            size=s['image_size_px']
            with Image.open(dst) as existing:
                if list(existing.size)!=size:raise ValueError('Previously verified image size changed')
        else:
            crop=pixels.crop(s)
            s['native_crop_sha256']=hashlib.sha256(crop.tobytes()).hexdigest()
            s['native_crop_shape']=list(crop.shape)
            size=(max(1,round(crop.shape[1]*1024/max(crop.shape))),
                  max(1,round(crop.shape[0]*1024/max(crop.shape))))
            image=Image.fromarray(stretch_uint8(crop)).resize(size,Image.Resampling.LANCZOS)
            if dst.exists():
                with Image.open(dst) as existing:
                    if existing.size!=image.size or not np.array_equal(np.asarray(existing),np.asarray(image)):
                        raise ValueError('Existing training pixels changed: '+str(dst))
            else:image.save(dst)
        s['image_sha256']=sha256_file(dst)
        if s['image_sha256'] in image_hash_splits:
            raise ValueError('Train crop duplicates protected evaluation image')
        link_or_copy(dst,roots['shp_control']/'images/train'/dst.name)
        s['image_size_px']=list(size)
        for arm,field in [('manual','manual_aabb_px'),('shp_control','shp_aabb_px')]:
            label=roots[arm]/'labels/train'/(s['stem']+'.txt')
            text=labels_text(s,field)
            if label.exists() and label.read_text(encoding='utf-8')!=text:
                raise ValueError('Existing training label changed')
            if not label.exists():
                label.write_text(text,encoding='utf-8')
            verify_labels(label)
        # Parallel class sidecars preserve SL/SP/FL/CL; main YAML is one-class Stage1.
        multi=out/'fourclass_label_sidecar'/ (s['stem']+'.txt')
        multi.parent.mkdir(exist_ok=True)
        multi.write_text(labels_text(s,'manual_aabb_px',True),encoding='utf-8')
        if i%100==0 or i+1==len(plan['samples']):
            print(f'Exported {i+1}/{len(plan["samples"])} native crops',flush=True)
    new_train=len(plan['samples'])
    for arm,root in roots.items():
        (root/'data.yaml').write_text(f'path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: LANDSLIDE\n',encoding='utf-8')
        # Exposure-matched historical comparison: same 4868 entries per epoch.
        # This is uniform deterministic resampling, not additional unique images.
        paths=[(root/'images/train'/(s['stem']+'.png')).as_posix() for s in plan['samples']]
        rng=np.random.default_rng(20261008)
        target=len((old/'train_stage1_rebalanced.txt').read_text(encoding='utf-8').splitlines())
        if new_train>target:
            raise ValueError('Exposure budget below unique train image count')
        indices=list(range(new_train))
        indices+=list(rng.permutation(new_train))*( (target-new_train)//new_train )
        indices+=list(rng.permutation(new_train))[:(target-len(indices))]
        rng.shuffle(indices)
        (root/'train_equal_exposure.txt').write_text('\n'.join(paths[i] for i in indices)+'\n',encoding='utf-8')
        (root/'data_equal_exposure.yaml').write_text(f'path: {root.as_posix()}\ntrain: train_equal_exposure.txt\nval: images/val\ntest: images/test\nnames:\n  0: LANDSLIDE\n',encoding='utf-8')
    old_yaml=out/'historical_baseline.yaml'
    old_yaml.write_text(f'path: {old.as_posix()}\ntrain: train_stage1_rebalanced.txt\nval: images/val\ntest: images/test\nnames:\n  0: LANDSLIDE\n',encoding='utf-8')
    labels=[a for s in plan['samples'] for a in s['labels']]
    input_keys={(s['tile'],s['source_id']) for s in plan['samples'] if s['kind']=='positive'}
    represented={(s['tile'],a['source_id']) for s in plan['samples'] for a in s['labels']}
    center_bins=Counter()
    for s in plan['samples']:
        if s['kind']=='positive':
            a=next(a for a in s['labels'] if a['source_id']==s['source_id'])
            xy=yolo_box(a['manual_aabb_px'],s['window_px'],s['native_crop_shape'][::-1])
            center_bins['center_10pct' if .45<=xy[0]<=.55 and .45<=xy[1]<=.55 else 'outside_center_10pct']+=1
    # Fully audit paired image identity and every exported label.
    for split in ('train','val','test'):
        for image in (roots['manual']/'images'/split).glob('*.png'):
            if sha256_file(image)!=sha256_file(roots['shp_control']/'images'/split/image.name):
                raise ValueError('Paired image mismatch')
            for root in roots.values():
                verify_labels(root/'labels'/split/(image.stem+'.txt'))
    train_hashes=Counter(s['image_sha256'] for s in plan['samples'])
    report=dict(status='complete_data_export_no_model_training',generated_utc=datetime.now(timezone.utc).isoformat(),
        input_positive_boxes=plan['input_positive_boxes'],input_background_boxes=plan['input_selected_background_boxes'],
        train_positive_images=len(input_keys),train_negative_images=new_train-len(input_keys),
        train_images=new_train,train_label_occurrences=len(labels),
        represented_unique_slide_ids=len({sid for _,sid in represented}),represented_tile_source_keys=len(represented),
        accepted_primary_keys=len(input_keys),class_label_occurrences=dict(Counter(a['class_name'] for a in labels)),
        unrepresented_confirmed_box_count=plan['input_positive_boxes']-len(represented),
        primary_center_distribution=dict(center_bins),duplicate_train_image_hash_groups=sum(n>1 for n in train_hashes.values()),
        held_primary_or_background_count=len(plan['held']),held_reason_counts=dict(Counter(h['reason'].split(':')[0] for h in plan['held'])),
        val_images=actual['val'],test_images=actual['test'],
        val_labels=sum(r['label_count'] for r in eval_records if r['split']=='val'),
        test_labels=sum(r['label_count'] for r in eval_records if r['split']=='test'),
        historical_summary=frozen['summary']['splits'],equal_exposure_train_entries=target,
        frozen_eval_hash_audit='all val/test image and label hashes match historical manifest',
        cross_split_image_hash_overlap=0,train_protected_tile_or_source_overlap=0,
        comparison='manual vs shp_control: identical images, source identities, negatives, split, exposure; labels alone vary. Historical vs new: holistic data change, not label-only causal evidence.',
        limitations=['Validation/test retain historical annotations; human-reviewed independent evaluation remains future work.',
                    'Catalog and manual extents protect known landslides; human confirmed background does not prove undiscovered slides absent.',
                    'Uniform exposure matching repeats images; it does not equalize unique image or target counts.',
                    'Fourclass train labels are a sidecar, not a full fourclass evaluation dataset.'],
        datasets={k:str(v/'data.yaml') for k,v in roots.items()},historical_baseline=str(old_yaml))
    write_json(out/'eval_manifest.json',eval_records)
    plan['status']='exported'
    write_json(out/'export_plan.json',plan)
    write_json(out/'build_report.json',report)
    protocol=dict(status='data_ready_training_not_started',architecture='YOLOv8n Stage1 LANDSLIDE',
        label_quality_comparison={k:str(v/'data_equal_exposure.yaml') for k,v in roots.items()},
        historical_comparison={'historical':str(old_yaml),'new':str(roots['manual']/'data_equal_exposure.yaml')},
        training=dict(model='yolov8n.pt',epochs=150,patience=0,imgsz=1024,batch=16,workers=8,
                      cache=False,scale=0.5,seeds=[0,1,2],resume=False,deterministic=True),
        exposure=dict(train_entries_per_epoch=target,epochs=150,
                      note='Pin identical optimizer and augmentation settings. Match dataloader lengths; audit actual optimizer steps and initialization SHA before running.'),
        evaluation=dict(conf_pool=.001,nms_iou=.7,match_iou=.5,
                        threshold_selection='validation only; candidate cap 2455; lock threshold before test'),
        note='Fresh train all arms under one current environment for strict comparison; historical reported scores are contextual references only.')
    write_json(out/'experiment_protocol.json',protocol)
    lines=[f'# 人工修框检测数据集 {out.name}', '',
           f'训练：{report["train_positive_images"]} 张正样本裁片 + {report["train_negative_images"]} 张空标签负样本裁片。',
           f'验证：{actual["val"]} 张；测试：{actual["test"]} 张；两者沿用旧固定划分和原标签。', '',
           'manual/data.yaml：人工修框；shp_control/data.yaml：同图同负样本的原 SHP 框对照。',
           f'data_equal_exposure.yaml 每轮 {target} 条训练输入，均匀重复采样，便于与旧重平衡臂匹配曝光。',
           'historical_baseline.yaml：旧数据参考，不修改旧数据。', '',
           '当前人工审核全部来自训练候选图幅，不能随机分到旧验证/测试集。',
           '不用等待数量相等；先做同图框质量对照，再做全量新数据与历史方案的整体对照。',
           '本次只导出数据，没有训练模型或产生新的性能指标。', '',
           'inputs/index.json 冻结输入及 SHA；export_plan.json 记录所有原像素坐标、邻近标签和暂扣原因。',
           'build_report.json 为数量与完整性检查；experiment_protocol.json 为待执行训练协议。',
           'fourclass_label_sidecar 是类别追溯标签，主数据为 Stage1 单类检测。',
           '人工排除、截断和未审核目标不会成为负样本。背景导出批次：'+str(plan['selected_negative_batches'])+'。', '',
           '验证/测试仍用历史标注，其边界误差会影响绝对指标。未来需单独审核固定评估图幅。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack',type=Path,default=PACK)
    ap.add_argument('--old',type=Path,default=OLD)
    ap.add_argument('--out',type=Path,default=BASE/'data/processed/manual_yolo_20261008')
    ap.add_argument('--plan-only',action='store_true')
    ap.add_argument('--resume-export',action='store_true',help='Resume a failed frozen export, verifying existing pixels and labels')
    args=ap.parse_args()
    args.pack,args.old,args.out=args.pack.resolve(),args.old.resolve(),args.out.resolve()
    if args.out.exists() and not args.resume_export:
        raise FileExistsError('Use a new output directory; immutable exports are never overwritten.')
    if args.resume_export:
        if read_json(args.out/'build_state.json')['status'] not in {'failed','building'}:
            raise ValueError('Only a failed or interrupted export can resume')
        plan=read_json(args.out/'export_plan.json')
        for item in plan['inputs'].values():
            if sha256_file(Path(item['snapshot']))!=item['sha256']:
                raise ValueError('Frozen input checksum changed')
    else:
        args.out.mkdir(parents=True)
    write_json(args.out/'build_state.json',dict(status='building'))
    try:
        if not args.resume_export:
            index=snapshot_inputs(args.pack,args.out)
            plan=make_plan(args.pack,args.out,index)
        fingerprint={str(Path(__file__).resolve()):sha256_file(Path(__file__))}
        for name in ('build_center_candidate_manifest.py','make_centered_review_pack.py',
                     'check_tile_connectivity.py','continue_ctx_review_download.py','manual_box_review.py'):
            script=Path(__file__).with_name(name)
            fingerprint[str(script.resolve())]=sha256_file(script)
        write_json(args.out/'generator_fingerprints.json',fingerprint)
        if not args.plan_only:
            export(plan,args.out,args.old)
        write_json(args.out/'build_state.json',dict(status='planned' if args.plan_only else 'complete'))
    except Exception as exc:
        write_json(args.out/'build_state.json',dict(status='failed',error=repr(exc)))
        raise


if __name__=='__main__':
    main()
