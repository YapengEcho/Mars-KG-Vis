"""Freeze catalog-wide human/SHP provenance and geographic splits, without pixels."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
import shapefile
import numpy as np
from make_centered_review_pack import read_geotransform,discover_tifs
from check_tile_connectivity import tiles_of_bbox

OUT=ROOT/'runs/research/catalog_expansion_20261024_v1'
DATA=ROOT/'data/processed/catalog_hybrid_20261024_v1'
K=3396190*math.pi/180
PX=4*K/47420
CAT=ROOT/'data/raw/mars_landslides.json'
SHP=ROOT/'data/raw/mars_landslide_boundaries/landslide_boundaries/火星滑坡-带边界.shp'
TYPES=['SL','SP','FL','CL']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def tile_gt(tile):
    e,n=tile.split('_');x=int(e[1:]);y=int(n[1:])
    return [x*K,PX,0.,(y+4)*K,0.,-PX]


def world(gt,b):
    xs=[gt[0]+b[i]*gt[1] for i in [0,2]];ys=[gt[3]+b[i]*gt[5] for i in [1,3]]
    return [min(xs),min(ys),max(xs),max(ys)]


def intersects(a,b):return min(a[2],b[2])-max(a[0],b[0])>1e-5 and min(a[3],b[3])-max(a[1],b[1])>1e-5


def touches(b):return sorted(tiles_of_bbox(b[0]/K+1e-9,b[1]/K+1e-9,b[2]/K-1e-9,b[3]/K-1e-9))


def union(boxes):return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]


def window(box,key):
    donors=touches(box);bounds=union([world(tile_gt(t),[0,0,47420,47420]) for t in donors])
    side=min(math.ceil(max(1024*PX,1.5*max(box[2]-box[0],box[3]-box[1]))/PX)*PX,bounds[2]-bounds[0],bounds[3]-bounds[1])
    assert side>=max(box[2]-box[0],box[3]-box[1])-1e-5,'Target exceeds source bounding rectangle'
    digest=hashlib.sha256(str(key).encode()).digest();fractions=[.2+.6*v/255 for v in digest[:2]]
    xy=[max(bounds[i],min(box[i]-(side-(box[i+2]-box[i]))*fractions[i],bounds[i+2]-side)) for i in [0,1]]
    result=[xy[0],xy[1],xy[0]+side,xy[1]+side]
    assert all(result[i]<=box[i]+1e-5 and result[i+2]>=box[i+2]-1e-5 for i in [0,1])
    return result,touches(result)


def main():
    assert not OUT.exists(),'Preserve frozen dataset design'
    catalog={int(r['source_id']):r for r in read(CAT)};pins={str(CAT):sha(CAT),str(Path(__file__)):sha(__file__)}
    for suffix in ['.shp','.shx','.dbf','.prj']:pins[str(SHP.with_suffix(suffix))]=sha(SHP.with_suffix(suffix))
    with shapefile.Reader(str(SHP),encoding='utf-8') as f:
        shapes={int(r.record[0]):dict(raw_type=r.record[1],bbox=[float(x)*K for x in r.shape.bbox]) for r in f.iterShapeRecords()}
    assert len(shapes)==len(catalog)==2971 and shapes.keys()==catalog.keys()
    for sid,r in shapes.items():
        assert {'RL':'SL','DF':'FL'}.get(r['raw_type'],r['raw_type'])==catalog[sid]['type']
    # Validate the inferred global grid against every available original header.
    originals=discover_tifs(ROOT/'data/raw/ctx_tiles');raw={}
    for tile,p in originals.items():
        gt,w,h=read_geotransform(p)
        assert [w,h]==[47420,47420] and np.allclose(gt,tile_gt(tile),rtol=0,atol=1e-6),tile
        raw[tile]=dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    confirmations=defaultdict(list);exclusions=defaultdict(list)
    for folder in sorted((ROOT/'data/processed').iterdir()):
        if not folder.is_dir():continue
        for filename in ['manual_boxes.reviewed.csv','cross_tile_manual_boxes.reviewed.csv']:
            p=folder/'manual_review'/filename
            if not p.is_file():continue
            pins[str(p)]=sha(p)
            with p.open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    sid=int(r['source_id'])
                    if sid not in catalog or r.get('confirmed','').lower()!='true' or r.get('label_scope')!='full_landslide':continue
                    if r.get('geotransform'):gt=json.loads(r['geotransform'])
                    elif r['tile'].startswith('E'):gt=tile_gt(r['tile'])
                    else:raise ValueError('Manual coordinate transform missing: '+str(p))
                    box=world(gt,[float(r[k]) for k in ['x0','y0','x1','y1']])
                    confirmations[sid].append(dict(path=str(p.resolve()),record=r,world_box_north_m=box,
                                                   priority=2 if r.get('geotransform') else 1))
        for filename in ['review_exclusions.json','cross_tile_exclusions.json']:
            p=folder/'manual_review'/filename
            if not p.is_file():continue
            pins[str(p)]=sha(p)
            for r in read(p).values():
                if str(r.get('source_id','')).isdigit():exclusions[int(r['source_id'])].append(dict(path=str(p.resolve()),record=r))
    aliases={};identity=ROOT/'data/processed/manual_eval_rebuild_20261011_v1/object_identity_adjudications.json'
    pins[str(identity)]=sha(identity)
    for r in read(identity)['adjudications'].values():
        if r['decision']=='same_physical_landslide' and r['confirmed_by']=='human_user':
            aliases[int(r['alias_key'].split(':')[-1])]=int(r['canonical_key'].split(':')[-1])
    forced={};forced_provenance={}
    # Preserve all historically protected evaluation geography. Test takes priority
    # when earlier protocols changed a region from validation to final evaluation.
    for filename in ['tile_split_plan_flval.json','tile_split_plan_flval_v2.json','tile_split_plan_flval_train_expansion.json']:
        p=ROOT/'data/raw'/filename;pins[str(p)]=sha(p)
        for tile,split in read(p).items():
            if split in ['val','test'] and forced.get(tile)!='test':forced[tile]=split;forced_provenance[tile]=str(p)
    p=ROOT/'data/raw/final_eval_tiles_frozen.json';pins[str(p)]=sha(p)
    for t in read(p)['tiles']:forced[t]='test';forced_provenance[t]=str(p)
    p=ROOT/'data/processed/manual_stage2_native_20261015_v1/manifest.json';pins[str(p)]=sha(p)
    native=read(p)
    training_tiles={s['tile'] for r in native['samples'] for s in r['sources']}
    assert not training_tiles&set(forced),'Existing training and protected geography overlap'
    for t in training_tiles:forced[t]='train';forced_provenance[t]=str(p)
    targets=[]
    for sid,c in sorted(catalog.items()):
        human=sorted(confirmations.get(sid,[]),key=lambda r:r['priority'],reverse=True)
        box=human[0]['world_box_north_m'] if human else shapes[sid]['bbox']
        row=dict(source_id=sid,canonical_source_id=aliases.get(sid,sid),catalog_type=c['type'],
                 raw_shp_type=shapes[sid]['raw_type'],shp_bbox_north_m=shapes[sid]['bbox'],
                 aabb_north_m=box,annotation_origin='human_full_extent' if human else 'shp_aabb_unreviewed',
                 expert_type_origin='original_catalog_unchanged',human_confirmation_records=human,
                 human_exclusion_records=exclusions.get(sid,[]),split=None,status='candidate',
                 future_review_needed=not bool(human),catalog_source_url='https://doi.org/10.13203/j.whugis20250300')
        if sid in aliases:row.update(status='alias_not_second_sample',future_review_needed=False)
        elif exclusions.get(sid):row.update(status='human_excluded_not_positive_not_background')
        else:
            b,donors=window(box,sid);row.update(window_north_m=b,source_tiles=donors)
            old={forced[t] for t in donors if t in forced}
            if len(old)>1:row['status']='held_crosses_protected_split_boundary'
        targets.append(row)
    # Join donor tiles only when compatible with preexisting geographic locks.
    parent={};locks={}
    def find(t):
        parent.setdefault(t,t)
        if parent[t]!=t:parent[t]=find(parent[t])
        return parent[t]
    for r in targets:
        if r['status']!='candidate':continue
        roots={find(t) for t in r['source_tiles']}
        old=set().union(*(locks.get(t,set()) for t in roots)) | {forced[t] for t in r['source_tiles'] if t in forced}
        if len(old)>1:r['status']='held_geographic_component_conflict';continue
        root=min(roots)
        for other in roots:parent[other]=root
        locks[root]=old
    groups=defaultdict(list)
    for r in targets:
        if r['status']=='candidate':groups[find(r['source_tiles'][0])].append(r)
    totals=Counter(r['catalog_type'] for r in targets if r['status']=='candidate')
    assigned={s:Counter() for s in ['train','val','test']};proportions={'train':.7,'val':.15,'test':.15};group_split={}
    for group,rows in groups.items():
        old={forced[t] for r in rows for t in r['source_tiles'] if t in forced}
        assert len(old)<=1
        if old:
            split=next(iter(old));group_split[group]=split;assigned[split].update(r['catalog_type'] for r in rows)
    free=[g for g in groups if g not in group_split]
    free.sort(key=lambda g:(-len(groups[g]),hashlib.sha256(g.encode()).hexdigest()))
    for group in free:
        counts=Counter(r['catalog_type'] for r in groups[group])
        def cost(split):
            return sum(((assigned[s][c]+(counts[c] if s==split else 0)-proportions[s]*totals[c])/max(totals[c],1))**2
                       for s in assigned for c in TYPES)
        split=min(['train','val','test'],key=cost);group_split[group]=split;assigned[split].update(counts)
    tile_split={}
    for group,rows in groups.items():
        for r in rows:
            r.update(split=group_split[group],spatial_group=group,status='planned_positive')
            for tile in r['source_tiles']:
                assert tile not in tile_split or tile_split[tile]==r['split']
                tile_split[tile]=r['split']
    # Label all intersecting known objects. Do not use unlabelled excluded/held
    # objects or opposite-split IDs as background inside any candidate image.
    canonical=[r for r in targets if r['source_id'] not in aliases]
    samples=[]
    for r in targets:
        if r['status']!='planned_positive':continue
        labels=[];blockers=[]
        for n in canonical:
            if not intersects(r['window_north_m'],n['aabb_north_m']):continue
            if n['human_exclusion_records'] or n.get('split')!=r['split']:
                blockers.append(n['source_id']);continue
            box=n['aabb_north_m'];w=r['window_north_m'];clipped=[max(box[0],w[0]),max(box[1],w[1]),min(box[2],w[2]),min(box[3],w[3])]
            labels.append(dict(source_id=n['source_id'],type=n['catalog_type'],annotation_origin=n['annotation_origin'],
                               aabb_north_m=clipped,truncated=clipped!=box))
        if blockers:r.update(status='held_ignored_or_other_split_neighbor',blocking_source_ids=blockers);continue
        assert any(l['source_id']==r['source_id'] and not l['truncated'] for l in labels)
        samples.append(dict(kind='positive',sample_id=f"slide_{r['source_id']:05d}",source_id=r['source_id'],
                            split=r['split'],spatial_group=r['spatial_group'],window_north_m=r['window_north_m'],
                            source_tiles=r['source_tiles'],annotation_origin=r['annotation_origin'],labels=labels))
    # Preserve existing confirmed background images, and recheck against the full
    # catalog before carrying them over. Never invent background from SHP absence.
    exp_path=ROOT/'data/processed/manual_yolo_20261009_v8/export_plan.json';pins[str(exp_path)]=sha(exp_path)
    previous=read(exp_path);negative_holds=[]
    for s in previous['samples']:
        if s['kind']!='negative':continue
        b=world(tile_gt(s['tile']),s['window_px']);hits=[r['source_id'] for r in canonical if intersects(b,r['aabb_north_m'])]
        if hits:negative_holds.append(dict(sample_id=s['stem'],reason='intersects_full_catalog_or_human_box',source_ids=hits));continue
        split=tile_split.get(s['tile'],forced.get(s['tile']))
        if split is None:negative_holds.append(dict(sample_id=s['stem'],reason='no_fixed_geographic_split'));continue
        p=ROOT/'data/processed/manual_yolo_20261009_v8/human/images/train'/f"{s['stem']}.png"
        if not p.exists():
            p=next((x for x in (ROOT/'data/processed/manual_yolo_20261009_v8').rglob(f"{s['stem']}.png")),None)
        assert p and p.is_file(),s['stem']
        assert sha(p)==s['image_sha256'];pins[str(p)]=s['image_sha256']
        samples.append(dict(kind='negative',sample_id=s['stem'],source_id=None,background_id=s['background_id'],
                            split=split,window_north_m=b,source_tiles=[s['tile']],annotation_origin='human_confirmed_background',
                            labels=[],copy_image=str(p.resolve()),image_sha256=s['image_sha256']))
    # Native retention rectangles include every catalog target (even excluded),
    # existing classifier crop, and confirmed background. This is separate from
    # permission to use an item in training.
    needed=defaultdict(list)
    def retain(b,reference):
        for tile in touches(b):
            gt=tile_gt(tile);full=world(gt,[0,0,47420,47420]);cut=[max(b[0],full[0]),max(b[1],full[1]),min(b[2],full[2]),min(b[3],full[3])]
            if cut[2]<=cut[0] or cut[3]<=cut[1]:continue
            px=[max(0,math.floor((cut[0]-gt[0])/PX)),max(0,math.floor((gt[3]-cut[3])/PX)),
                min(47420,math.ceil((cut[2]-gt[0])/PX)),min(47420,math.ceil((gt[3]-cut[1])/PX))]
            needed[tile].append(dict(window_px=px,reference=reference))
    for r in targets:
        b=r.get('window_north_m')
        if b is None:b,_=window(r['aabb_north_m'],'retain'+str(r['source_id']))
        retain(b,'catalog:'+str(r['source_id']))
    for r in native['samples']:
        b=r['native_window_world_m'];retain([b[0],-b[3],b[2],-b[1]],'old_classifier:'+str(r['source_id']))
    for s in samples:
        if s['kind']=='negative':retain(s['window_north_m'],'background:'+s['sample_id'])
    # Keep existing editor windows as raw pixels when retiring their original.
    for folder in (ROOT/'data/processed').iterdir():
        if not folder.is_dir():continue
        for p in (folder/'tiles').glob('*/centered_review_manifest.csv'):
            with p.open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    if not r.get('tile','').startswith('E'):continue
                    b=json.loads(r['window_origin']);retain(world(tile_gt(r['tile']),b),'review:'+str(p)+':'+str(r['source_id']))
    manifest=read(ROOT/'data/raw/ctx_tiles_manifest.json');download={r['tile']:r for r in manifest}
    pins[str(ROOT/'data/raw/ctx_tiles_manifest.json')]=sha(ROOT/'data/raw/ctx_tiles_manifest.json')
    counts={s:dict(Counter(r['annotation_origin'] for r in samples if r['split']==s and r['kind']=='positive')) for s in assigned}
    status=dict(Counter(r['status'] for r in targets));all_needed=sorted(needed)
    plan=dict(schema_version=1,version='catalog_hybrid_20261024_v1',role='new_version_original_experiments_remain_frozen',
              created_utc=datetime.now(timezone.utc).isoformat(),world_coordinates='Mars spherical equirectangular metres; x east, y north; source latitude interpretation unchanged',
              expert_type_labels_unchanged=True,raw_prj_not_used_for_earth_transform=True,targets=targets,samples=samples,
              target_status_counts=status,annotation_counts_by_split=counts,tile_split=tile_split,legacy_geography_locks=forced,
              ratios_requested=proportions,split_policy='same donor tile and canonical target cannot cross splits; retain old protected areas; conflicting windows held',
              aliases=aliases,negative_holds=negative_holds,input_sha256=pins,existing_raw_tiles=raw,
              native_retention_windows=dict(needed),download_rows={t:download[t] for t in all_needed if t in download},
              missing_download_metadata=[t for t in all_needed if t not in download and t not in raw],
              dataset=str(DATA),training_started=False,pixels_exported=False,
              evaluation_policy='SHP validation/test remains provisional; report human and SHP reference strata separately; do not compare a changed evaluation set as model improvement',
              future_review_policy='Join by stable source_id/canonical_source_id; human edits create a new annotation version, never silently replace frozen labels')
    OUT.mkdir(parents=True);DATA.mkdir(parents=True)
    write(OUT/'plan.json',plan);(OUT/'plan.sha256').write_text(sha(OUT/'plan.json')+'\n')
    write(DATA/'annotation_registry.json',dict(version=plan['version'],targets=targets,aliases=aliases,
                                             plan_sha256=sha(OUT/'plan.json'),training_started=False))
    with (DATA/'future_review_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['source_id','canonical_source_id','catalog_type','annotation_origin','future_review_needed','status','split','source_tiles']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in targets:writer.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if k=='source_tiles' else r.get(k) for k in fields})
    print(json.dumps(dict(targets=len(targets),status_counts=status,positive_annotation_counts_by_split=counts,
                          background_samples=sum(s['kind']=='negative' for s in samples),native_tiles=len(needed),
                          existing_raw_tiles=len(raw),missing_download_metadata=len(plan['missing_download_metadata'])),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
