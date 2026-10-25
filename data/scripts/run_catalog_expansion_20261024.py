"""Finite catalog exporter: preserve native regions, retire whole tiles, roll downloads."""
from collections import defaultdict
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'data/scripts')]
import numpy as np
from PIL import Image
import tifffile
import prepare_catalog_expansion_20261024 as P
from fetch_v2_train_tiles import download_zip,extract_verified,existing_tif
from build_native_classification_dataset import NativeReader,histogram_lut

DESIGN=P.OUT
OUT=ROOT/'runs/maintenance/catalog_expansion_20261024_v1'
NATIVE=ROOT/'data/processed/catalog_native_regions_20261024_v1'
SCRATCH=Path(tempfile.gettempdir())/'MarsKG-catalog-expansion-20261024'
CTX=SCRATCH/'ctx'
DATA=P.DATA
TASK='MarsKG-catalog-expansion-20261024-v1'
DELETE=ROOT/'tools/maintenance/delete_catalog_preserved_tile_20261024.ps1'


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for i in range(10):
        try:os.replace(tmp,path);return
        except PermissionError:
            if i==9:raise
            time.sleep(.1*(i+1))


def cells(required,side=8192):
    """Bound each region to one <=64MiB decoded native block; never downsample."""
    grouped={}
    for entry in required:
        x0,y0,x1,y1=entry['window_px']
        for gy in range(y0//side,(y1-1)//side+1):
            for gx in range(x0//side,(x1-1)//side+1):
                cut=[max(x0,gx*side),max(y0,gy*side),min(x1,(gx+1)*side),min(y1,(gy+1)*side)]
                key=gx,gy;grouped[key]=P.union([grouped[key],cut]) if key in grouped else cut
    return [list(map(int,grouped[k])) for k in sorted(grouped,key=lambda k:(k[1],k[0]))]


def preserve_region(raw,gt,box,dest):
    """Readback every decoded strip and compare to native source, not just headers."""
    x0,y0,x1,y1=box;h,w=y1-y0,x1-x0
    temporary=dest.with_suffix('.partial.tif')
    assert not temporary.exists() and not dest.exists(),'Preserve unreceipted cache files'
    tifffile.imwrite(temporary,raw[y0:y1,x0:x1],compression='deflate',predictor=2,rowsperstrip=128,
                     maxworkers=2,metadata=None,photometric='minisblack',
                     extratags=[(33550,'d',3,(gt[1],-gt[5],0.),False),
                                (33922,'d',6,(0.,0.,0.,gt[0]+x0*gt[1],gt[3]+y0*gt[5],0.),False)])
    digest=hashlib.sha256();row=0
    with tifffile.TiffFile(temporary) as tf:
        page=tf.pages[0];assert list(page.shape)==[h,w] and page.dtype==np.uint8
        for segment,index,shape in page.segments(maxworkers=2):
            assert index[2]==row
            n=min(segment.shape[1],h-row);value=segment[0,:n,:w,0]
            assert np.array_equal(value,raw[y0+row:y0+row+n,x0:x1]),'Decoded cache differs from original pixels'
            digest.update(np.ascontiguousarray(value).tobytes());row+=n
    assert row==h
    os.replace(temporary,dest)
    return dict(path=str(dest.resolve()),bytes=dest.stat().st_size,mtime_ns=dest.stat().st_mtime_ns,
                sha256=P.sha(dest),native_pixels_sha256=digest.hexdigest(),decoded_pixels_verified=True,
                window_px=box,size_px=[w,h],geotransform=[gt[0]+x0*gt[1],gt[1],0.,gt[3]+y0*gt[5],0.,gt[5]])


def prepare():
    assert not (OUT/'worker_protocol.json').exists(),'Preserve worker protocol'
    plan=P.read(DESIGN/'plan.json');assert P.sha(DESIGN/'plan.json')==(DESIGN/'plan.sha256').read_text().strip()
    required=defaultdict(list,{t:list(v) for t,v in plan['native_retention_windows'].items()})
    latest=ROOT/'runs/research/latest_detector_roi_preparation_20261024_v1/input_plan.json'
    lp=P.read(latest)
    for c in lp['cases']:
        required[c['tile']].append(dict(window_px=c['window_px'],reference='development_ROI:'+str(c['index'])))
    blocks={t:cells(v) for t,v in required.items()}
    for tile,entries in required.items():
        for entry in entries:
            b=entry['window_px']
            # Every requested pixel rectangle is covered by the per-cell bounds.
            covered=sum(max(0,min(b[2],r[2])-max(b[0],r[0]))*max(0,min(b[3],r[3])-max(b[1],r[1])) for r in blocks[tile])
            assert covered==(b[2]-b[0])*(b[3]-b[1]),tile
    used={t for s in plan['samples'] if s['kind']=='positive' for t in s['source_tiles']} | set(lp['sources'])
    local=list(plan['existing_raw_tiles'])
    local.sort(key=lambda t:(sum((r[2]-r[0])*(r[3]-r[1]) for r in blocks[t]),t))
    # Process locally available originals first to reclaim disk before expansion.
    following=sorted(used-set(local),key=lambda t:(t not in lp['sources'],t))
    pins={str(p):P.sha(p) for p in [DESIGN/'plan.json',latest,Path(__file__),DELETE,
                                    ROOT/'data/scripts/start_catalog_expansion_20261024.ps1',Path(P.__file__)]}
    for p,d in plan['input_sha256'].items():assert P.sha(p)==d,p
    spec=dict(status='locked_before_smoke',created_utc=utc(),files_sha256=pins,queue=local+following,
              blocks=blocks,required_window_count={t:len(v) for t,v in required.items()},
              initial_raw_tiles=len(local),finite_tiles=len(local+following),
              minimum_D_free_bytes=1<<30,minimum_C_free_bytes=6<<30,
              original_deletion='Only after stripwise pixel equality, cache SHA256, coverage proof and PowerShell revalidation',
              no_unreviewed_negative_samples=True,expert_types_unchanged=True,periodic_triggers=False,
              training_started=False,final_test_used_for_training=False)
    OUT.mkdir(parents=True,exist_ok=True);NATIVE.mkdir(parents=True,exist_ok=True)
    write(OUT/'worker_protocol.json',spec);(OUT/'worker_protocol.sha256').write_text(P.sha(OUT/'worker_protocol.json')+'\n')
    write(OUT/'worker_status.json',dict(status='prepared_not_started',finite_tiles=len(spec['queue']),completed_tiles=0,
                                      cached_tiles=0,deleted_original_bytes=0,exported_images=0))
    print('PREPARED_FINITE_CATALOG_WORKER',len(spec['queue']),'tiles;',len(local),'existing originals first',flush=True)


def check():
    spec=P.read(OUT/'worker_protocol.json')
    assert P.sha(OUT/'worker_protocol.json')==(OUT/'worker_protocol.sha256').read_text().strip()
    for p,d in spec['files_sha256'].items():assert P.sha(p)==d,p
    return spec,P.read(DESIGN/'plan.json')


def smoke():
    spec,_=check();dest=OUT/'cache_smoke';assert not dest.exists();dest.mkdir()
    raw=np.random.default_rng(20261024).integers(0,256,(285,413),dtype=np.uint8)
    region=preserve_region(raw,[0.,5.,0.,1000.,0.,-5.],[13,21,399,279],dest/'region.tif')
    assert region['decoded_pixels_verified']
    assert tifffile.imread(dest/'region.tif').shape==(258,386)
    # A multi-cell rectangle must be preserved exactly, without double counting.
    b=[7,11,17190,9199];rs=cells([dict(window_px=b)])
    assert sum((r[2]-r[0])*(r[3]-r[1]) for r in rs)==(b[2]-b[0])*(b[3]-b[1])
    write(OUT/'readiness.json',dict(status='passed_actual_deflate_strip_readback_and_coverage_smoke',
                                    protocol_sha256=P.sha(OUT/'worker_protocol.json'),test_region=region))
    print('PASSED_NATIVE_CACHE_SMOKE',flush=True)


def cache_sources():
    sources=defaultdict(list)
    for p in NATIVE.glob('*/manifest.json'):
        d=P.read(p)
        if d.get('status')!='verified_complete_native_regions':continue
        for r in d['entries']:
            sources[d['tile']].append(dict(tile=d['tile'],path=r['path'],bytes=r['bytes'],mtime_ns=r['mtime_ns'],
                                           geotransform=r['geotransform'],size_px=r['size_px'],cache_sha256=r['sha256']))
    return sources


def preserve_tile(tile,path,spec,plan):
    folder=NATIVE/tile;folder.mkdir(exist_ok=True);manifest=folder/'manifest.json'
    if manifest.exists():
        previous=P.read(manifest)
        if previous.get('status')=='verified_complete_native_regions':return previous
    else:previous=dict(status='preserving',tile=tile,entries=[],source_url=plan['download_rows'][tile]['url'])
    gt,w,h=P.read_geotransform(path);assert [w,h]==[47420,47420] and np.allclose(gt,P.tile_gt(tile),atol=1e-6,rtol=0)
    raw=tifffile.memmap(path)
    try:
        for index,box in enumerate(spec['blocks'][tile]):
            if index<len(previous['entries']):
                r=previous['entries'][index];assert P.sha(r['path'])==r['sha256'] and r['window_px']==box;continue
            assert shutil.disk_usage(NATIVE).free>spec['minimum_D_free_bytes']+(box[2]-box[0])*(box[3]-box[1])*1.02,'D reserve reached'
            r=preserve_region(raw,gt,box,folder/f'block_{index:03d}.tif')
            previous['entries'].append(r);write(manifest,previous)
        previous.update(status='verified_complete_native_regions',original=dict(path=str(path.resolve()),bytes=path.stat().st_size,
                                                                               sha256=P.sha(path)),
                        required_windows_covered=True,required_window_count=spec['required_window_count'][tile],
                        coverage_plan_sha256=P.sha(OUT/'worker_protocol.json'),completed_utc=utc())
        write(manifest,previous)
    finally:raw._mmap.close()
    return previous


def retire(tile,manifest):
    original=Path(manifest['original']['path'])
    if not original.exists():return 0
    receipt=OUT/'retirement_receipts'/f'{tile}.json'
    if not receipt.exists():
        write(receipt,dict(status='verified_native_regions_preserved',tile=tile,original=manifest['original'],
                           cache_manifest=str(NATIVE/tile/'manifest.json'),cache_manifest_sha256=P.sha(NATIVE/tile/'manifest.json'),
                           caches=manifest['entries'],required_windows_covered=True))
    result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(DELETE),'-Receipt',str(receipt)],
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,result.stderr
    return manifest['original']['bytes']


def local_original(tile,plan):
    p=Path(plan['existing_raw_tiles'].get(tile,{}).get('path','__missing__'))
    if p.is_file():return p
    found=existing_tif(CTX,tile)
    return Path(found['path']) if found else None


def export_positive(sample,sources,reader,spec):
    receipt=DATA/'receipts'/f"{sample['sample_id']}.json"
    if receipt.exists():
        saved=P.read(receipt)
        for p,d in saved.get('files_sha256',{}).items():assert P.sha(p)==d,p
        return saved
    if not all(t in sources for t in sample['source_tiles']):return None
    b=sample['window_north_m'];px=P.PX
    gx0=math.floor(b[0]/px+1e-7);gx1=math.ceil(b[2]/px-1e-7)
    gy0=math.floor(-b[3]/px+1e-7);gy1=math.ceil(-b[1]/px-1e-7)
    w,h=gx1-gx0,gy1-gy0;gt=[gx0*px,px,0.,-gy0*px,0.,-px]
    assert shutil.disk_usage(SCRATCH).free>2*w*h+spec['minimum_C_free_bytes'],'C crop reserve reached'
    token=uuid.uuid4().hex;raw_path=SCRATCH/(token+'_raw.bin');stretch_path=SCRATCH/(token+'_stretch.bin')
    raw=np.memmap(raw_path,mode='w+',shape=(h,w),dtype=np.uint8);stretched=None;image=None
    counts=np.zeros(256,dtype=np.int64);digest=hashlib.sha256()
    entry=dict(geotransform=gt,size_px=[w,h],sources=[r for t in sample['source_tiles'] for r in sources[t]])
    try:
        for start,block,used in reader.slabs(entry):
            raw[start:start+len(block)]=block;counts+=np.bincount(block.ravel(),minlength=256);digest.update(block.tobytes())
        raw.flush();lut,contrast=histogram_lut(counts)
        if counts[0]/(w*h)>.25:
            result=dict(status='held_native_nodata_fraction',sample_id=sample['sample_id'],zero_fraction=float(counts[0]/(w*h)),files_sha256={})
            write(receipt,result);return result
        stretched=np.memmap(stretch_path,mode='w+',shape=(h,w),dtype=np.uint8)
        for start in range(0,h,128):stretched[start:start+128]=lut[raw[start:start+128]]
        stretched.flush();image=Image.frombuffer('L',(w,h),stretched,'raw','L',0,1)
        split=sample['split'];stem=sample['sample_id'];paths={}
        detector=DATA/f'detector/images/{split}/{stem}.png';detector.parent.mkdir(parents=True,exist_ok=True)
        image.resize((1024,1024),Image.Resampling.BILINEAR).save(detector);paths[str(detector)]=P.sha(detector)
        type_name=next(l['type'] for l in sample['labels'] if l['source_id']==sample['source_id'])
        for size in [224,448]:
            dest=DATA/f'classification{size}/{split}/{type_name}/{stem}.png';dest.parent.mkdir(parents=True,exist_ok=True)
            image.resize((size,size),Image.Resampling.BILINEAR).save(dest);paths[str(dest)]=P.sha(dest)
        labels=[];sidecar=[]
        for label in sample['labels']:
            v=label['aabb_north_m'];x0=(v[0]-gt[0])/(w*px);x1=(v[2]-gt[0])/(w*px)
            y0=(gt[3]-v[3])/(h*px);y1=(gt[3]-v[1])/(h*px)
            x0,y0,x1,y1=np.clip([x0,y0,x1,y1],0,1)
            assert x0<x1 and y0<y1
            labels.append(f'0 {(x0+x1)/2:.10f} {(y0+y1)/2:.10f} {x1-x0:.10f} {y1-y0:.10f}')
            sidecar.append(dict(label,normalized_xyxy=[float(v) for v in [x0,y0,x1,y1]]))
        label_path=DATA/f'detector/labels/{split}/{stem}.txt';label_path.parent.mkdir(parents=True,exist_ok=True)
        label_path.write_text('\n'.join(labels)+'\n');paths[str(label_path)]=P.sha(label_path)
        result=dict(status='exported',sample_id=stem,source_id=sample['source_id'],split=split,
                    annotation_origin=sample['annotation_origin'],labels=sidecar,geotransform=gt,native_size_px=[w,h],
                    native_pixels_sha256=digest.hexdigest(),contrast=contrast,files_sha256=paths,
                    source_tiles=sample['source_tiles'],plan_sha256=P.sha(DESIGN/'plan.json'))
        write(receipt,result);return result
    finally:
        if image is not None:image.close()
        if stretched is not None:stretched._mmap.close()
        raw._mmap.close()
        for p in [raw_path,stretch_path]:
            assert p.parent.resolve()==SCRATCH.resolve()
            p.unlink(missing_ok=True)


def export_background(sample):
    receipt=DATA/'receipts'/f"{sample['sample_id']}.json"
    if receipt.exists():return P.read(receipt)
    assert P.sha(sample['copy_image'])==sample['image_sha256']
    dest=DATA/f"detector/images/{sample['split']}/{sample['sample_id']}.png";dest.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(sample['copy_image'],dest)
    except OSError:shutil.copy2(sample['copy_image'],dest)
    label=DATA/f"detector/labels/{sample['split']}/{sample['sample_id']}.txt";label.parent.mkdir(parents=True,exist_ok=True);label.write_text('')
    result=dict(status='exported',sample_id=sample['sample_id'],source_id=None,split=sample['split'],labels=[],
                annotation_origin='human_confirmed_background',source_tiles=sample['source_tiles'],
                files_sha256={str(dest):P.sha(dest),str(label):P.sha(label)},plan_sha256=P.sha(DESIGN/'plan.json'))
    write(receipt,result);return result


def export_ready(plan,spec,state,reader):
    sources=cache_sources();completed=0;held=0
    for sample in plan['samples']:
        if shutil.disk_usage(DATA).free<spec['minimum_D_free_bytes']+(100<<20):break
        r=export_background(sample) if sample['kind']=='negative' else export_positive(sample,sources,reader,spec)
        if r is not None:
            if r['status']=='exported':completed+=1
            else:held+=1
        if completed%50==0:
            state.update(exported_images=completed,held_export_quality=held);write(OUT/'worker_status.json',state)
    state.update(exported_images=completed,held_export_quality=held)
    write(OUT/'worker_status.json',state)


def run():
    spec,plan=check();assert P.read(OUT/'readiness.json')['protocol_sha256']==P.sha(OUT/'worker_protocol.json')
    state=P.read(OUT/'worker_status.json');assert state['status'] in ['prepared_not_started','failed_resumable','paused_by_user']
    SCRATCH.mkdir(parents=True,exist_ok=True);CTX.mkdir(exist_ok=True);(DATA/'receipts').mkdir(exist_ok=True)
    state.update(status='running',pid=os.getpid(),started_utc=state.get('started_utc',utc()),phase='preserving_existing_originals')
    write(OUT/'worker_status.json',state)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    reader=NativeReader(budget=512<<20)
    try:
        for index,tile in enumerate(spec['queue']):
            state.update(current_tile=tile,tile_index=index+1,phase='native_preservation');write(OUT/'worker_status.json',state)
            m=NATIVE/tile/'manifest.json'
            if m.exists() and P.read(m).get('status')=='verified_complete_native_regions':
                manifest=P.read(m)
            else:
                path=local_original(tile,plan)
                if path is None:
                    assert shutil.disk_usage(CTX).free>2*plan['download_rows'][tile]['size_bytes']+spec['minimum_C_free_bytes'],'C download reserve reached'
                    state.update(phase='downloading_one_tile');write(OUT/'worker_status.json',state)
                    # The separate finite6 restoration owns those tiles until it
                    # has validated their historical native windows.
                    restoration=OUT/'eval_restore_status.json'
                    if restoration.exists():
                        rs=P.read(restoration)
                        while tile in P.read(ROOT/'runs/experiments/detector_classifier_bridge_20261015_v2/input_plan.json')['sources'] and rs['status']=='running':
                            state.update(phase='waiting_for_owned_eval_tile');write(OUT/'worker_status.json',state);time.sleep(10);rs=P.read(restoration)
                        path=local_original(tile,plan)
                    if path is None:
                        zipped=download_zip(plan['download_rows'][tile],CTX,4);found=extract_verified(plan['download_rows'][tile],zipped,CTX);path=Path(found['path'])
                manifest=preserve_tile(tile,path,spec,plan)
            state['deleted_original_bytes']+=retire(tile,manifest)
            zipped=CTX/plan['download_rows'][tile]['zip']
            if zipped.is_file():
                assert zipped.resolve().parent==CTX.resolve() and not zipped.is_symlink();zipped.unlink()
            state.update(completed_tiles=index+1,cached_tiles=len(list(NATIVE.glob('*/manifest.json'))),
                         free_D_bytes=shutil.disk_usage(DATA).free,phase='preservation_completed')
            write(OUT/'worker_status.json',state)
            print('PRESERVED_AND_RETIRED',tile,index+1,'/',len(spec['queue']),'freeD_GiB',round(state['free_D_bytes']/2**30,2),flush=True)
            if (index+1)%5==0 or index+1==len(spec['queue']):export_ready(plan,spec,state,reader)
        export_ready(plan,spec,state,reader)
        receipts=[P.read(p) for p in (DATA/'receipts').glob('*.json')]
        assert len(receipts)==len(plan['samples']),'Some planned samples lack native sources; preserve partial export'
        yaml=f'path: {str(DATA/"detector").replace(chr(92),"/")}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: landslide\n'
        (DATA/'detector/data.yaml').write_text(yaml,encoding='utf-8')
        write(DATA/'manifest.json',dict(status='exported_requires_independent_validation_before_training',
                                       plan_sha256=P.sha(DESIGN/'plan.json'),receipts=receipts,training_started=False))
        state.update(status='completed',phase='complete_not_training',finished_utc=utc());write(OUT/'worker_status.json',state)
    except BaseException as error:
        state.update(status='failed_resumable',error=repr(error),updated_utc=utc());write(OUT/'worker_status.json',state);raise
    finally:
        while reader.cache:reader.evict()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='smoke':smoke()
    elif mode=='run':
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'worker.log').open('a',encoding='utf-8',buffering=1) as stream:
            sys.stdout=stream;sys.stderr=stream;run()
    else:raise ValueError(mode)
