"""Recover held crops without editing human decisions or a frozen dataset."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import shapefile

from export_manual_detection_dataset import (BASE, OLD, Pixels, union, read_json,
    write_json, link_or_copy, labels_text, verify_labels, stretch_uint8)
from build_center_candidate_manifest import load_overrides, contains, TYPE_ALIASES, sha256_file
from manual_box_review import has_complete_geometry
from make_centered_review_pack import _record_for_tile, _tile_bounds, map_source_records, _intersects
from check_tile_connectivity import tiles_of_bbox


def frozen_records(plan, tile):
    idx=plan['inputs']
    with Path(idx['queue']['snapshot']).open(encoding='utf-8-sig') as f:
        queue={(r['tile'],int(r['source_id'])):r for r in csv.DictReader(f)}
    manual=load_overrides(Path(idx['overrides']['snapshot']),set(queue))
    exclusions=read_json(Path(idx['exclusions']['snapshot']))
    geometry=read_json(Path(idx['geometry_holds']['snapshot']))
    with shapefile.Reader(shp=idx['shp']['snapshot'],shx=idx['shx']['snapshot'],dbf=idx['dbf']['snapshot']) as sf:
        sources=map_source_records(sf.shapeRecords())
    t=plan['tile_sources'][tile];gt=t['geotransform'];width,height=t['tile_size_px']
    records={}
    for sid,source in sources.items():
        r=_record_for_tile(sid,source,gt,_tile_bounds(gt,width,height))
        if r is None:continue
        k=(tile,sid);o=manual.get(k);e=exclusions.get(f'{tile}:{sid}');boxes=[r['aabb_px']]
        if o:boxes.append(o['aabb_px'])
        if e and e.get('exclusion_box_px'):boxes.append(e['exclusion_box_px'])
        reason='human_excluded' if e else 'protected_source' if tiles_of_bbox(*source.shape.bbox)&set(plan['protected_tiles']) else 'incomplete_reference' if k not in queue or not has_complete_geometry(queue[k]) or f'{tile}:{sid}' in geometry else 'missing_confirmed_manual' if not o else 'visible_core' if o['label_scope']!='full_landslide' else None
        r.update(support=union(boxes),manual_box=o['aabb_px'] if o else None,hold_reason=reason,
                 class_name=TYPE_ALIASES.get(r['raw_type'],r['raw_type']))
        records[sid]=r
    return records


def verify_window(records,window,primary):
    hits=[r for r in records.values() if _intersects(tuple(window),tuple(r['support']))]
    if primary not in {r['source_id'] for r in hits}:raise ValueError('Missing primary')
    if any(r['hold_reason'] or not contains(window,r['support']) for r in hits):
        raise ValueError('Crop contains incomplete or held neighbor')
    return sorted(hits,key=lambda r:r['source_id'])


def recover(base,out):
    base,out=base.resolve(),out.resolve()
    if out.exists():raise FileExistsError(out)
    old_plan=read_json(base/'export_plan.json');plan=copy.deepcopy(old_plan)
    screen=read_json(base/'held_recrop_geometry_screen.json')
    candidates=[r for r in screen['candidates_and_holds'] if r['geometry_recrop_candidate']]
    # Images remain immutable shared pixels; copy other files so labels and
    # manifests cannot mutate the source dataset through a hard link.
    def clone(source,target):
        if 'images' in Path(source).parts:link_or_copy(Path(source),Path(target))
        else:shutil.copy2(source,target)
        return str(target)
    shutil.copytree(base,out,copy_function=clone)
    write_json(out/'build_state.json',{'status':'building_recovery'})
    for key,item in plan['inputs'].items():
        item['snapshot']=str(out/Path(item['snapshot']).relative_to(base))
    write_json(out/'inputs/index.json',plan['inputs'])
    records_by_tile={};pixels=Pixels();added=[];hashes={s['image_sha256'] for s in plan['samples']}
    eval_hashes={sha for e in read_json(out/'eval_manifest.json') for path,sha in e['hashes'].items() if path.startswith('images/')}
    for candidate in candidates:
        tile,sid=candidate['tile'],candidate['source_id'];w=candidate['geometry_recrop_candidate']['window_px']
        records=records_by_tile.setdefault(tile,frozen_records(plan,tile))
        hits=verify_window(records,w,sid)
        src=copy.deepcopy(plan['tile_sources'][tile])
        if src['kind']=='cached_context':
            cache=read_json(Path(src['manifest']['snapshot']));entries=[e for e in cache['entries'] if contains(e['window_origin'],w)]
            if not entries:raise ValueError('Missing native context')
            src=dict(kind='cached_context',**min(entries,key=lambda e:e['shape'][0]*e['shape'][1]))
        s=dict(kind='positive',tile=tile,source_id=sid,window_px=w,pixel_source=src,
            stem=f'{tile}_recovered_{sid:05d}',labels=[dict(source_id=r['source_id'],class_name=r['class_name'],manual_aabb_px=r['manual_box'],shp_aabb_px=r['aabb_px']) for r in hits],recovery_origin='alternate_offset_same_full_boxes')
        crop=pixels.crop(s);s['native_crop_sha256']=hashlib.sha256(crop.tobytes()).hexdigest();s['native_crop_shape']=list(crop.shape)
        size=(round(crop.shape[1]*1024/max(crop.shape)),round(crop.shape[0]*1024/max(crop.shape)))
        img=Image.fromarray(stretch_uint8(crop)).resize(size,Image.Resampling.LANCZOS)
        image=out/'manual/images/train'/(s['stem']+'.png');img.save(image);s['image_sha256']=sha256_file(image);s['image_size_px']=list(size)
        if s['image_sha256'] in hashes|eval_hashes:raise ValueError('New crop duplicates existing image')
        hashes.add(s['image_sha256']);link_or_copy(image,out/'shp_control/images/train'/image.name)
        for arm,field in [('manual','manual_aabb_px'),('shp_control','shp_aabb_px')]:
            label=out/arm/'labels/train'/(s['stem']+'.txt');label.write_text(labels_text(s,field),encoding='utf-8');verify_labels(label)
        (out/'fourclass_label_sidecar'/(s['stem']+'.txt')).write_text(labels_text(s,'manual_aabb_px',True),encoding='utf-8')
        plan['samples'].append(s);added.append(s)
    n=len(plan['samples']);target=4868
    for arm in ['manual','shp_control']:
        root=out/arm;paths=[(root/'images/train'/(s['stem']+'.png')).as_posix() for s in plan['samples']]
        rng=np.random.default_rng(20261008);indices=list(range(n));indices+=list(rng.permutation(n))*((target-n)//n);indices+=list(rng.permutation(n))[:target-len(indices)];rng.shuffle(indices)
        (root/'train_equal_exposure.txt').write_text('\n'.join(paths[i] for i in indices)+'\n',encoding='utf-8')
        for equal in [False,True]:
            (root/('data_equal_exposure.yaml' if equal else 'data.yaml')).write_text(f'path: {root.as_posix()}\ntrain: '+('train_equal_exposure.txt' if equal else 'images/train')+'\nval: images/val\ntest: images/test\nnames:\n  0: LANDSLIDE\n',encoding='utf-8')
    represented={(s['tile'],a['source_id']) for s in plan['samples'] for a in s['labels']}
    remaining=[r for r in read_json(base/'unrepresented_confirmed_boxes.json') if (r['tile'],r['source_id']) not in represented]
    write_json(out/'unrepresented_confirmed_boxes.json',remaining)
    plan.update(status='exported_recovery',recovery_base=str(base),recovery_added_image_count=len(added));write_json(out/'export_plan.json',plan)
    report=read_json(base/'build_report.json');labels=[a for s in plan['samples'] for a in s['labels']]
    report.update(train_positive_images=sum(s['kind']=='positive' for s in plan['samples']),train_images=n,train_label_occurrences=len(labels),represented_unique_slide_ids=len({sid for _,sid in represented}),represented_tile_source_keys=len(represented),accepted_primary_keys=sum(s['kind']=='positive' for s in plan['samples']),class_label_occurrences=dict(Counter(a['class_name'] for a in labels)),unrepresented_confirmed_box_count=len(remaining),recovery_base=str(base),recovered_images=len(added),recovered_distinct_slides=len(read_json(base/'unrepresented_confirmed_boxes.json'))-len(remaining),datasets={a:str(out/a/'data.yaml') for a in ['manual','shp_control']},historical_baseline=str(out/'historical_baseline.yaml'),primary_center_distribution='see parent report for original crops; additions use alternative offsets',held_primary_or_background_count='see parent hold list and current unrepresented_confirmed_boxes.json',held_reason_counts='parent report retained in parent_build_report.json')
    shutil.copy2(base/'build_report.json',out/'parent_build_report.json');write_json(out/'build_report.json',report)
    protocol=read_json(out/'experiment_protocol.json');protocol['label_quality_comparison']={a:str(out/a/'data_equal_exposure.yaml') for a in ['manual','shp_control']};protocol['historical_comparison']={'historical':str(out/'historical_baseline.yaml'),'new':str(out/'manual/data_equal_exposure.yaml')};write_json(out/'experiment_protocol.json',protocol)
    # These follow-up reports describe the parent triage; preserve that distinction.
    for name in ['data_comparison.json','historical_eval_identity_recheck.json','spatial_guard_audit.json','held_neighbor_followup.json','held_recrop_geometry_screen.json']:
        if (out/name).exists():(out/name).rename(out/('parent_'+name))
    sheet=Image.new('RGB',(1024,math_ceil_rows(len(added))*280),(16,16,16));draw=ImageDraw.Draw(sheet)
    for i,s in enumerate(added):
        img=Image.open(out/'manual/images/train'/(s['stem']+'.png')).convert('RGB');img.thumbnail((252,252));d=ImageDraw.Draw(img)
        for line in labels_text(s,'manual_aabb_px').splitlines():
            _,cx,cy,w,h=map(float,line.split());d.rectangle(((cx-w/2)*img.width,(cy-h/2)*img.height,(cx+w/2)*img.width,(cy+h/2)*img.height),outline=(80,255,130),width=2)
        x=(i%4)*256;y=(i//4)*280;sheet.paste(img,(x,y+24));draw.text((x+3,y+3),s['tile']+' #'+str(s['source_id']),fill='white')
    sheet.save(out/'recovery_qa.png');write_json(out/'recovery_additions.json',added)
    shutil.copy2(Path(__file__),out/'code_snapshot'/Path(__file__).name)
    write_json(out/'recovery_generator.json',{'path':str(Path(__file__).resolve()),'sha256':sha256_file(Path(__file__))})
    (out/'README.md').write_text(f'# 人工框数据恢复版\n\n基于 {base.name}，新增 {len(added)} 张重裁图，恢复 {report["recovered_distinct_slides"]} 个滑坡。\n训练 {n} 张（正 {report["train_positive_images"]}，负 {report["train_negative_images"]}），覆盖 {report["represented_unique_slide_ids"]} 个滑坡，余 {len(remaining)} 个暂扣。\n验证615张、测试497张及标签保持原冻结版本。\nmanual 与 shp_control 为同图框对照；data_equal_exposure.yaml 每轮4868条。\n未训练模型。原人工修订与排除记录未改。\nparent_*报告属于上一版，当前统计以 build_report.json 为准。\n',encoding='utf-8')
    (out/'comparison.md').write_text((out/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
    write_json(out/'build_state.json',{'status':'awaiting_visual_and_spatial_check'})
    print(json.dumps({k:report[k] for k in ['train_positive_images','train_negative_images','train_images','represented_unique_slide_ids','recovered_distinct_slides','unrepresented_confirmed_box_count']},ensure_ascii=False))


def math_ceil_rows(n):return (n+3)//4


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,default=BASE/'data/processed/manual_yolo_20261008_v2');ap.add_argument('--out',type=Path,default=BASE/'data/processed/manual_yolo_20261008_v3');a=ap.parse_args();recover(a.base,a.out)
