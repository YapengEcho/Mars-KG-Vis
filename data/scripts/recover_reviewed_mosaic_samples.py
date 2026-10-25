"""Freeze cross-tile reviews and recover complete crops from native CTX pixels.

World rectangles use x east, y south, in metres. Unknown/excluded neighbors,
missing pixels and protected evaluation geography remain hard holds.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from export_manual_detection_dataset import (BASE, OLD, read_json, write_json,
    union, positive_window, contains, labels_text, verify_labels, stretch_uint8,
    link_or_copy, inventory)
from build_center_candidate_manifest import TYPE_ALIASES, sha256_file
from make_centered_review_pack import discover_tifs, read_geotransform, _intersects
from manual_box_review import has_complete_geometry
from cross_tile_neighbor_review import PACK, load_sources, compose
from check_tile_connectivity import load_tile_geo, tiles_of_bbox_overlap

DATASET=BASE/'data/processed/manual_yolo_20261008_v4'
SCREEN=BASE/'runs/mosaic_recovery_20261008'
OUT=BASE/'data/processed/manual_yolo_20261008_v5'
K=3396190*math.pi/180


def world_box(gt,box):
    x0,y0,x1,y1=box
    return [gt[0]+x0*gt[1],-gt[3]-y0*gt[5],gt[0]+x1*gt[1],-gt[3]-y1*gt[5]]


def pixel_box(gt,box):
    return [(box[0]-gt[0])/gt[1],(box[1]+gt[3])/-gt[5],
            (box[2]-gt[0])/gt[1],(box[3]+gt[3])/-gt[5]]


def intersects(a,b):return _intersects(tuple(a),tuple(b))


def freeze_single(pack,folder):
    if folder.exists():raise FileExistsError(folder)
    paths={'manifest':pack/'mosaic_review_manifest.json',
           'manual':pack/'manual_review/cross_tile_manual_boxes.reviewed.csv',
           'exclusions':pack/'manual_review/cross_tile_exclusions.json'}
    if not paths['manifest'].exists():raise FileNotFoundError(paths['manifest'])
    for _ in range(3):
        present={key:path.exists() for key,path in paths.items()}
        data={key:path.read_bytes() if present[key] else b'{}' if key=='exclusions' else b'tile,source_id\n' for key,path in paths.items()}
        if all(path.exists()==present[key] and (not present[key] or path.read_bytes()==data[key]) for key,path in paths.items()):break
    else:raise ValueError('Reviews changed during snapshot')
    folder.mkdir(parents=True);index={}
    for key,path in paths.items():
        target=folder/path.name;target.write_bytes(data[key])
        index[key]=dict(source=str(path),source_exists=present[key],snapshot=str(target),sha256=hashlib.sha256(data[key]).hexdigest())
    write_json(folder/'index.json',index)
    return index


def freeze(pack,folder):
    index=freeze_single(pack,folder);registry=pack/'review_followup_packs.json'
    if not registry.exists():return index
    registry_bytes=registry.read_bytes();packs=json.loads(registry_bytes)['packs'];groups=[index]
    for i,path in enumerate(packs,1):
        extra=freeze_single(Path(path),folder/f'followup_{i}');groups.append(extra)
        index.update({f'followup_{i}_{key}':value for key,value in extra.items()})
    manifests=[read_json(Path(group['manifest']['snapshot'])) for group in groups]
    entries={};pending={};rows=[];exclusions={};fields=[]
    for group,manifest in zip(groups,manifests):
        for entry in manifest['entries']:
            sid=entry['source_id']
            if sid in entries and entries[sid]!=entry:raise ValueError('Conflicting review manifests for source '+str(sid))
            entries[sid]=entry
        for entry in manifest['pending']:pending[entry['source_id']]=entry
        with Path(group['manual']['snapshot']).open(encoding='utf-8-sig') as f:
            reader=csv.DictReader(f);fields=list(dict.fromkeys(fields+(reader.fieldnames or [])));rows+=list(reader)
        for key,value in read_json(Path(group['exclusions']['snapshot'])).items():
            if key in exclusions and exclusions[key]!=value:raise ValueError('Conflicting exclusions')
            exclusions[key]=value
    if len({(row['tile'],int(row['source_id'])) for row in rows})!=len(rows):raise ValueError('Duplicate manual source across review packs')
    for value in index.values():
        source=Path(value['source'])
        if source.exists()!=value['source_exists'] or (value['source_exists'] and sha256_file(source)!=value['sha256']):raise ValueError('Review packs changed while merging; take a fresh snapshot')
    if registry.read_bytes()!=registry_bytes:raise ValueError('Review pack registry changed during snapshot')
    registry_snapshot=folder/'review_followup_packs.json';registry_snapshot.write_bytes(registry_bytes)
    index['followup_registry']=dict(source=str(registry),source_exists=True,snapshot=str(registry_snapshot),sha256=hashlib.sha256(registry_bytes).hexdigest())
    merged=dict(manifests[0],entries=list(entries.values()),pending=[row for sid,row in pending.items() if sid not in entries],ready_count=len(entries),pending_count=len(set(pending)-set(entries)))
    for key,filename in [('manifest','combined_mosaic_review_manifest.json'),('manual','combined_mosaic_manual_boxes.reviewed.csv'),('exclusions','combined_mosaic_exclusions.json')]:
        index['primary_'+key]=dict(index[key]);target=folder/filename
        if key=='manual':
            with target.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        else:write_json(target,merged if key=='manifest' else exclusions)
        index[key]=dict(source='derived coherent merge of pinned primary and followup files',snapshot=str(target),sha256=sha256_file(target))
    write_json(folder/'index.json',index)
    return index


def relocate_review_inputs(index,source_folder,target_folder):
    result=copy.deepcopy(index)
    for value in result.values():value['snapshot']=str(target_folder/Path(value['snapshot']).relative_to(source_folder))
    return result


def records_from_reviews(plan,sources,index):
    queue,manual,excluded,geometry,_,_,protected=inventory(plan['inputs'],range(1,20))
    decisions={};supports={};main_excluded=set()
    for (tile,sid),row in manual.items():
        gt=plan['tile_sources'][tile]['geotransform'];box=world_box(gt,row['aabb_px'])
        supports.setdefault(sid,[]).append(box)
        if (tile,sid) in queue and has_complete_geometry(queue[tile,sid]) and f'{tile}:{sid}' not in geometry and row['label_scope']=='full_landslide':
            if sid in decisions and max(abs(a-b) for a,b in zip(decisions[sid],box))>.05:
                raise ValueError('Conflicting original complete manual boxes: '+str(sid))
            decisions[sid]=box
    for key,row in excluded.items():
        tile,sid_text=key.rsplit(':',1);sid=int(sid_text);main_excluded.add(sid)
        if row.get('exclusion_box_px'):
            supports.setdefault(sid,[]).append(world_box(plan['tile_sources'][tile]['geotransform'],row['exclusion_box_px']))
    manifest=read_json(Path(index['manifest']['snapshot']));entries={r['source_id']:r for r in manifest['entries']}
    mosaic_excluded=read_json(Path(index['exclusions']['snapshot']))
    mosaic_manual={}
    with Path(index['manual']['snapshot']).open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if f"{row['tile']}:{row['source_id']}" in mosaic_excluded:continue
            sid=int(row['source_id']);entry=entries[sid];gt=json.loads(row['geotransform'])
            if row['tile']!=entry['tile'] or gt!=entry['geotransform'] or row['confirmed'].lower()!='true' or row['label_scope']!='full_landslide' or not row['reviewer'].strip() or not row['reason'].strip():
                raise ValueError('Invalid mosaic confirmation: '+str(sid))
            box=[float(row[v]) for v in ['x0','y0','x1','y1']]
            if not contains([0,0,*entry['size_px']],box):raise ValueError('Mosaic box outside reviewed window')
            w=world_box(gt,box);saved=json.loads(row['world_box_m'])
            if max(abs(a-b) for a,b in zip(w,[saved[0],-saved[1],saved[2],-saved[3]]))>1e-6:raise ValueError('Saved mosaic world transform mismatch')
            if sid in main_excluded or sid in decisions:raise ValueError('New review conflicts with original complete decision: '+str(sid))
            decisions[sid]=w;mosaic_manual[sid]=row;supports.setdefault(sid,[]).append(w)
    excluded_ids=set(main_excluded)
    for row in mosaic_excluded.values():
        sid=int(row['source_id'])
        if row['negative_eligible'] is not False or row['training_use']!='exclude_positive':raise ValueError('Invalid mosaic exclusion')
        if sid in mosaic_manual or sid in decisions:raise ValueError('Conflicting exclusion: '+str(sid))
        excluded_ids.add(sid);supports.setdefault(sid,[]).append(world_box(row['geotransform'],row['exclusion_box_px']))
    records={}
    for sid,source in sources.items():
        lo,la,hi,ha=source.shape.bbox;shp=[lo*K,-ha*K,hi*K,-la*K]
        reason='human_excluded_not_background' if sid in excluded_ids else 'protected_eval_source' if sid in plan['protected_sources'] else 'unreviewed_or_incomplete' if sid not in decisions else None
        records[sid]=dict(source_id=sid,aabb_px=shp,manual_box=decisions.get(sid),
            support=union([shp]+supports.get(sid,[])),hold_reason=reason,
            class_name=TYPE_ALIASES.get(str(source.record[1]).strip(),str(source.record[1]).strip()))
    return records,entries,mosaic_manual,mosaic_excluded


def native_sources(plan):
    pool=[]
    for tile,path in sorted(discover_tifs(BASE/'data/raw/ctx_tiles').items()):
        if tile in plan['protected_tiles']:continue
        gt,w,h=read_geotransform(path)
        pool.append(dict(tile=tile,path=str(path.resolve()),bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,geotransform=list(gt),size_px=[w,h]))
    for tile,src in plan['tile_sources'].items():
        if src['kind']!='cached_context':continue
        gt=src['geotransform'];cache=read_json(Path(src['manifest']['snapshot']))
        for entry in cache['entries']:
            if entry['stride']!=1:raise ValueError('Native cache was downsampled')
            x,y,_,_=entry['window_origin'];path=Path(entry['path']);h,w=entry['shape']
            pool.append(dict(tile=tile,path=str(path.resolve()),bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,geotransform=[gt[0]+x*gt[1],gt[1],0,gt[3]+y*gt[5],0,gt[5]],size_px=[w,h],cache_sha256=entry['sha256']))
    return pool


def coverage_rows(entry,start,end):
    """Exact native-grid coverage, chunked to avoid a large second full bitmap."""
    gt=entry['geotransform'];width,_=entry['size_px']
    xx=gt[0]+np.arange(width)*gt[1];yy=gt[3]+np.arange(start,end)*gt[5]
    mask=np.zeros((end-start,width),dtype=bool)
    for src in entry['sources']:
        t=src['geotransform'];w,h=src['size_px']
        sx=np.rint((xx-t[0])/t[1]).astype(np.int64);sy=np.rint((yy-t[3])/t[5]).astype(np.int64)
        ix=np.flatnonzero((sx>=0)&(sx<w));iy=np.flatnonzero((sy>=0)&(sy<h))
        mask[np.ix_(iy,ix)]=True
    return mask


def exact_coverage(entry):
    # Each source covers a rectangle of output grid indices. Check the exact
    # union of those rectangles, including any one-pixel gap at a seam.
    gt=entry['geotransform'];w,h=entry['size_px'];rects=[]
    xx=gt[0]+np.arange(w)*gt[1];yy=gt[3]+np.arange(h)*gt[5]
    for src in entry['sources']:
        t=src['geotransform'];sw,sh=src['size_px']
        sx=np.rint((xx-t[0])/t[1]).astype(np.int64);sy=np.rint((yy-t[3])/t[5]).astype(np.int64)
        ix=np.flatnonzero((sx>=0)&(sx<sw));iy=np.flatnonzero((sy>=0)&(sy<sh))
        if len(ix) and len(iy):rects.append([int(ix[0]),int(iy[0]),int(ix[-1])+1,int(iy[-1])+1])
    edges=sorted({0,h}|{r[j] for r in rects for j in [1,3]})
    for y0,y1 in zip(edges,edges[1:]):
        if y0==y1:continue
        intervals=sorted((r[0],r[2]) for r in rects if r[1]<=y0 and r[3]>=y1);end=0
        for a,b in intervals:
            if a>end:return False
            end=max(end,b)
        if end<w:return False
    return True


def native_crop(entry):
    """Read every native output pixel; previews are never used for training."""
    w,h=entry['size_px'];crop=np.empty((h,w),dtype=np.uint8)
    for src in entry['sources']:
        path=Path(src['path'])
        if path.stat().st_size!=src['bytes'] or path.stat().st_mtime_ns!=src['mtime_ns']:raise ValueError('Native source changed')
        if src.get('cache_sha256') and sha256_file(path)!=src['cache_sha256']:raise ValueError('Native context changed')
    for y in range(0,h,128):
        end=min(y+128,h);gt=entry['geotransform'];slab=dict(entry,size_px=[w,end-y],geotransform=[gt[0],gt[1],0,gt[3]+y*gt[5],0,gt[5]])
        block,mask,stride=compose(slab,max_side=max(w,end-y))
        if stride!=1 or not mask.all():raise ValueError('Native mosaic coverage gap')
        crop[y:end]=block
    return crop


def native_hash(crop):
    digest=hashlib.sha256()
    for y in range(0,crop.shape[0],128):digest.update(crop[y:y+128].tobytes())
    return digest.hexdigest()


def eval_geo(plan):
    protected=set(plan['protected_tiles']);geo=load_tile_geo(BASE/'data/raw/ctx_tiles',protected)
    for tile in protected-set(geo):
        lon,lat=map(int,re.fullmatch(r'E(-?\d+)_N(-?\d+)',tile).groups())
        geo[tile]=([lon*K,4*K/47420,0,(lat+4)*K,0,-4*K/47420],47420,47420)
    return geo


def protected_overlap(box,geo):
    return tiles_of_bbox_overlap([box[0]/K,-box[3]/K,box[2]/K,-box[1]/K],geo)


def hits_and_hold(records,box):
    hits=sorted((r for r in records.values() if intersects(box,r['support'])),key=lambda r:r['source_id'])
    invalid=[r for r in hits if r['hold_reason'] or not contains(box,r['support'])]
    return hits,invalid


def tight_closure(primary,records):
    window=[math.floor(records[primary]['support'][0]),math.floor(records[primary]['support'][1]),
            math.ceil(records[primary]['support'][2]),math.ceil(records[primary]['support'][3])]
    for _ in range(len(records)+1):
        hits=[r for r in records.values() if intersects(window,r['support'])]
        blocked=[r for r in hits if r['hold_reason']]
        if blocked:return None,[], 'neighbor_hold:'+','.join(f"{r['source_id']}={r['hold_reason']}" for r in blocked)
        hull=union([window]+[r['support'] for r in hits])
        expanded=[math.floor(hull[0]),math.floor(hull[1]),math.ceil(hull[2]),math.ceil(hull[3])]
        if expanded==window:return window,sorted(hits,key=lambda r:r['source_id']),None
        window=expanded
    raise ValueError('Closure did not converge')


def make_screen(base,folder,index):
    plan,sources=load_sources(base);records,entries,manual,excluded=records_from_reviews(plan,sources,index)
    pool=native_sources(plan);geo=eval_geo(plan);retained=[];removed=[];updates=[]
    for sample in plan['samples']:
        gt=sample.get('geotransform',plan['tile_sources'].get(sample['tile'],{}).get('geotransform'));box=world_box(gt,sample['window_px'])
        hits,invalid=hits_and_hold(records,box)
        if sample['kind']=='negative':
            if hits:removed.append(dict(stem=sample['stem'],kind='negative',background_id=sample['background_id'],reason='updated_world_extent_intersection',source_ids=[r['source_id'] for r in hits]));continue
        elif invalid:
            removed.append(dict(stem=sample['stem'],kind='positive',tile=sample['tile'],source_id=sample['source_id'],reason='updated_neighbor_hold',source_ids=[r['source_id'] for r in invalid]));continue
        else:
            newlabels=[dict(source_id=r['source_id'],class_name=r['class_name'],manual_aabb_px=pixel_box(gt,r['manual_box']),shp_aabb_px=pixel_box(gt,r['aabb_px'])) for r in hits]
            if not labels_equivalent(sample['labels'],newlabels):
                sample=dict(sample,labels=newlabels);updates.append(sample['stem'])
        retained.append(sample)
    candidates=[dict(tile=r['tile'],source_id=r['source_id']) for r in read_json(base/'unrepresented_confirmed_boxes.json')]
    candidates += [dict(tile=r['tile'],source_id=r['source_id']) for r in removed if r['kind']=='positive']
    represented={l['source_id'] for s in retained for l in s['labels']}
    candidates += [dict(tile=entries[sid]['anchor_tile'],source_id=sid) for sid in manual if sid not in represented and not any(c['source_id']==sid for c in candidates)]
    seen=set();additions=[];held=[]
    for c in candidates:
        sid=c['source_id'];key=(c['tile'],sid)
        if key in seen:continue
        seen.add(key)
        if records[sid]['hold_reason']:
            held.append(dict(c,reason=records[sid]['hold_reason']));continue
        gt=plan['tile_sources'].get(c['tile'],{}).get('geotransform')
        if gt is None:gt=entries[sid]['geotransform']
        # Work in this anchor's native pixel grid, with all global known extents.
        local={i:dict(r,aabb_px=pixel_box(gt,r['aabb_px']),manual_box=pixel_box(gt,r['manual_box']) if r['manual_box'] else None,support=pixel_box(gt,r['support'])) for i,r in records.items()}
        reason=None;chosen=None
        for margin in [1.5,1.25,1.1,1.0]:
            if margin==1.0:window,hits,reason=tight_closure(sid,local)
            else:window,hits,reason=positive_window(sid,local,[-1e8,-1e8,1e8,1e8],f"{c['tile']}:{sid}:mosaic",margin)
            if reason:continue
            window=[math.floor(window[0]),math.floor(window[1]),math.ceil(window[2]),math.ceil(window[3])]
            world=world_box(gt,window)
            if protected_overlap(world,geo):reason='protected_eval_geography';continue
            srcs=[s for s in pool if intersects(world,world_box(s['geotransform'],[0,0,*s['size_px']]))]
            if not srcs:reason='native_sources_missing';continue
            if any(abs(s['geotransform'][1]/gt[1]-1)>.01 or abs(s['geotransform'][5]/gt[5]-1)>.01 for s in srcs):raise ValueError('Mosaic resolution mismatch')
            entry=dict(geotransform=[gt[0]+window[0]*gt[1],gt[1],0,gt[3]+window[1]*gt[5],0,gt[5]],size_px=[window[2]-window[0],window[3]-window[1]],sources=srcs)
            if not exact_coverage(entry):reason='native_coverage_gap';continue
            chosen=dict(kind='positive',tile=c['tile'],source_id=sid,window_px=window,
                pixel_source=dict(kind='native_mosaic',**entry),labels=[dict(source_id=r['source_id'],class_name=r['class_name'],manual_aabb_px=r['manual_box'],shp_aabb_px=r['aabb_px']) for r in hits],
                stem=f"{c['tile']}_mosaic_{sid:05d}",geotransform=gt,recovery_origin='human_reviewed_complete_cross_tile_neighbors',world_window_m=world)
            break
        if chosen:additions.append(chosen)
        else:held.append(dict(c,reason=reason))
    value=dict(status='screened_not_exported',base=str(base),review_inputs=index,
        reviewed_accepted=len(manual),reviewed_excluded=len(excluded),ready=read_json(Path(index['manifest']['snapshot']))['ready_count'],
        retained=retained,removed=removed,updated_existing_labels=updates,additions=additions,held=held)
    write_json(folder/'screen.json',value)
    print(json.dumps(dict(accepted=len(manual),excluded=len(excluded),retained=len(retained),removed_by_kind=dict(Counter(r['kind'] for r in removed)),candidate_additions=len(additions),candidate_holds=len(held),held_reasons=dict(Counter(r['reason'].split(':')[0] for r in held))),ensure_ascii=False),flush=True)
    return value


def labels_equivalent(a,b):
    a={r['source_id']:r for r in a};b={r['source_id']:r for r in b}
    if set(a)!=set(b):return False
    return all(a[sid]['class_name']==b[sid]['class_name'] and
        all(max(abs(x-y) for x,y in zip(a[sid][key],b[sid][key]))<1e-7 for key in ['manual_aabb_px','shp_aabb_px']) for sid in a)


def export_screen(base,folder,out,resume=False):
    if out.exists():
        if not resume or read_json(out/'build_state.json')['status']!='building_mosaic_recovery':raise FileExistsError(out)
        original_index=read_json(folder/'inputs/index.json');prior_index=read_json(out/'mosaic_review_inputs/index.json')
        if {k:v['sha256'] for k,v in original_index.items()}!={k:v['sha256'] for k,v in prior_index.items()}:raise ValueError('Resume review snapshot changed')
    screen=read_json(folder/'screen.json');oldplan=read_json(base/'export_plan.json')
    if screen['base']!=str(base):raise ValueError('Screen base mismatch')
    def clone(src,dst):
        if 'images' in Path(src).parts:link_or_copy(Path(src),Path(dst))
        else:shutil.copy2(src,dst)
        return str(dst)
    if not out.exists():
        shutil.copytree(base,out,copy_function=clone);write_json(out/'build_state.json',dict(status='building_mosaic_recovery'))
        shutil.copytree(folder/'inputs',out/'mosaic_review_inputs',dirs_exist_ok=True)
    plan=copy.deepcopy(oldplan)
    for value in plan['inputs'].values():value['snapshot']=str(out/Path(value['snapshot']).relative_to(base))
    write_json(out/'inputs/index.json',plan['inputs'])
    index=relocate_review_inputs(screen['review_inputs'],folder/'inputs',out/'mosaic_review_inputs')
    write_json(out/'mosaic_review_inputs/index.json',index)
    _,sources=load_sources(out);records,_,_,_=records_from_reviews(plan,sources,index)
    for removed in screen['removed']:
        for arm in ['manual','shp_control']:
            for sub,suffix in [('images','.png'),('labels','.txt')]:
                path=(out/arm/sub/'train'/(removed['stem']+suffix)).resolve()
                if not path.is_relative_to(out.resolve()):raise ValueError('Unsafe removal')
                if path.exists():path.unlink()
        sidecar=out/'fourclass_label_sidecar'/(removed['stem']+'.txt')
        if sidecar.exists():sidecar.unlink()
    samples=screen['retained'];image_hashes={s['image_sha256'] for s in samples}
    eval_hashes={digest for row in read_json(out/'eval_manifest.json') for path,digest in row['hashes'].items() if path.startswith('images/')}
    actual_added=[];duplicates=[]
    for sample in screen['additions']:
        crop=native_crop(sample['pixel_source']);sample['native_crop_sha256']=native_hash(crop);sample['native_crop_shape']=list(crop.shape)
        size=(round(crop.shape[1]*1024/max(crop.shape)),round(crop.shape[0]*1024/max(crop.shape)))
        img=Image.fromarray(stretch_uint8(crop)).resize(size,Image.Resampling.LANCZOS);del crop
        path=out/'manual/images/train'/(sample['stem']+'.png')
        if (base/'manual/images/train'/path.name).exists():raise ValueError('New stem conflicts with frozen parent')
        temporary=path.with_suffix('.tmp');img.save(temporary,format='PNG');temporary.replace(path)
        digest=sha256_file(path)
        if digest in eval_hashes:raise ValueError('New image duplicates evaluation')
        if digest in image_hashes:path.unlink();duplicates.append(sample['stem']);continue
        image_hashes.add(digest);sample.update(image_sha256=digest,image_size_px=list(size))
        link_or_copy(path,out/'shp_control/images/train'/path.name);actual_added.append(sample)
        print('RECOVERED_NATIVE',sample['stem'],sample['native_crop_shape'],flush=True)
    samples+=actual_added
    for sample in samples:
        for arm,field in [('manual','manual_aabb_px'),('shp_control','shp_aabb_px')]:
            path=out/arm/'labels/train'/(sample['stem']+'.txt');path.write_text(labels_text(sample,field),encoding='utf-8');verify_labels(path)
        (out/'fourclass_label_sidecar'/(sample['stem']+'.txt')).write_text(labels_text(sample,'manual_aabb_px',True),encoding='utf-8')
    n=len(samples);rng=np.random.default_rng(20261008);indices=list(range(n))
    while len(indices)<4868:indices+=list(rng.permutation(n))[:4868-len(indices)]
    if n>4868:raise ValueError('Exposure protocol must be revisited')
    rng.shuffle(indices)
    for arm in ['manual','shp_control']:
        root=out/arm;paths=[(root/'images/train'/(s['stem']+'.png')).as_posix() for s in samples]
        (root/'train_equal_exposure.txt').write_text('\n'.join(paths[i] for i in indices)+'\n',encoding='utf-8')
        for equal in [False,True]:
            (root/('data_equal_exposure.yaml' if equal else 'data.yaml')).write_text(f'path: {root.as_posix()}\ntrain: '+('train_equal_exposure.txt' if equal else 'images/train')+'\nval: images/val\ntest: images/test\nnames:\n  0: LANDSLIDE\n',encoding='utf-8')
    represented={label['source_id'] for s in samples for label in s['labels']}
    remaining=[r for r in read_json(base/'unrepresented_confirmed_boxes.json') if r['source_id'] not in represented]
    original_manual=inventory(plan['inputs'],range(1,20))[1]
    for (tile,sid),row in original_manual.items():
        if sid not in represented and not any(r['source_id']==sid for r in remaining):remaining.append(dict(tile=tile,source_id=sid,manual_aabb_px=row['aabb_px'],reason='updated_world_neighbor_hold'))
    plan.update(samples=samples,status='exported_mosaic_recovery',parent_dataset=str(base),mosaic_review_inputs=index,mosaic_recovery=True,added_mosaic_images=len(actual_added),removed_existing=screen['removed'],parent_plan_sha256=sha256_file(base/'export_plan.json'))
    if (base/'mosaic_recovery_validation.json').exists():plan['parent_native_validation_sha256']=sha256_file(base/'mosaic_recovery_validation.json')
    report=read_json(base/'build_report.json');labels=[label for s in samples for label in s['labels']]
    report.update(status='awaiting_mosaic_validation',train_positive_images=sum(s['kind']=='positive' for s in samples),train_negative_images=sum(s['kind']=='negative' for s in samples),train_images=n,train_label_occurrences=len(labels),represented_unique_slide_ids=len(represented),represented_tile_source_keys=len({(s['tile'],l['source_id']) for s in samples for l in s['labels']}),accepted_primary_keys=sum(s['kind']=='positive' for s in samples),unrepresented_confirmed_box_count=len(remaining),class_label_occurrences=dict(Counter(l['class_name'] for l in labels)),datasets={a:str(out/a/'data.yaml') for a in ['manual','shp_control']},historical_baseline=str(out/'historical_baseline.yaml'),reviewed_mosaic_positive_boxes=screen['reviewed_accepted'],reviewed_mosaic_exclusions=screen['reviewed_excluded'],new_mosaic_images=len(actual_added),removed_existing_by_kind=dict(Counter(r['kind'] for r in screen['removed'])),recovered_prior_held=len(read_json(base/'unrepresented_confirmed_boxes.json'))-len(remaining),mosaic_duplicate_crops=duplicates,primary_center_distribution='offset crop windows; see export_plan.json',held_reason_counts=dict(Counter(r['reason'].split(':')[0] for r in screen['held'])))
    write_json(out/'export_plan.json',plan);write_json(out/'build_report.json',report)
    write_json(out/'unrepresented_confirmed_boxes.json',remaining);write_json(out/'mosaic_additions.json',actual_added);write_json(out/'mosaic_recovery_screen.json',screen)
    parent_version=base.name.rsplit('_',1)[-1]
    for name in ['recovery_validation.json','mosaic_recovery_validation.json','recovery_additions.json','background_completion_qa.png','experiment_protocol.json','comparison.md','README.md','live_input_postcheck.json']:
        if (out/name).exists():(out/name).rename(out/(f'parent_{parent_version}_'+name))
    shutil.copy2(Path(__file__),out/'code_snapshot'/Path(__file__).name)
    text=f'# 人工框与跨图幅恢复 {out.name.rsplit("_",1)[-1]}\n\n训练 {n} 张：正 {report["train_positive_images"]}、负 {report["train_negative_images"]}。本版新增 {len(actual_added)} 张原生像素拼接裁片，覆盖 {len(represented)} 个滑坡。\n\n跨图幅审核累计 {screen["reviewed_accepted"]} 个确认、{screen["reviewed_excluded"]} 个排除，决定独立冻结。训练裁片中所有已知滑坡均完整标注；排除目标不当作背景。负样本重新检查扩大后的边界。\n\nmanual与shp_control图像、目标ID、负样本和4868条曝光列表相同，只变更框。验证615张、测试497张保持原冻结评估，不共享已知滑坡与图幅。\n\n{parent_version} 与旧人工记录保留。尚未进行正式模型训练。parent_* 属于上一版本。当前统计以 build_report.json 为准。\n'
    (out/'README.md').write_text(text,encoding='utf-8');(out/'comparison.md').write_text(text,encoding='utf-8')
    sheet=Image.new('RGB',(1024,max(1,math.ceil(min(len(actual_added),40)/4))*280),(12,12,12));draw=ImageDraw.Draw(sheet)
    for i,s in enumerate(actual_added[:40]):
        img=Image.open(out/'manual/images/train'/(s['stem']+'.png')).convert('RGB');img.thumbnail((252,252));d=ImageDraw.Draw(img)
        for line in labels_text(s,'manual_aabb_px').splitlines():
            _,cx,cy,w,h=map(float,line.split());d.rectangle(((cx-w/2)*img.width,(cy-h/2)*img.height,(cx+w/2)*img.width,(cy+h/2)*img.height),outline=(70,255,130),width=2)
        x=i%4*256;y=i//4*280;sheet.paste(img,(x,y+24));draw.text((x+2,y+3),s['stem'],fill='white')
    sheet.save(out/'mosaic_recovery_qa.png')
    print(json.dumps({k:report[k] for k in ['train_images','train_positive_images','train_negative_images','new_mosaic_images','represented_unique_slide_ids','recovered_prior_held','unrepresented_confirmed_box_count']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,default=DATASET);ap.add_argument('--screen',type=Path,default=SCREEN);ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--export',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.export:export_screen(a.base.resolve(),a.screen.resolve(),a.out.resolve(),a.resume)
    else:
        index=freeze(PACK,a.screen.resolve()/'inputs');make_screen(a.base.resolve(),a.screen.resolve(),index)
