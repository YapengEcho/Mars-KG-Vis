"""Independent review of complete cross-tile neighbors; never edit old decisions.

Native TIFFs remain in one documented CTX cache (D by default). Previews
are nearest-sampled mosaics on an anchor tile's 5 m grid, with at most half a
source pixel registration rounding. Saved boxes include the mosaic transform.
"""
from __future__ import annotations
import argparse
import csv
from functools import lru_cache
from http.server import ThreadingHTTPServer
import io
import json
import math
import shutil
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import shapefile
import tifffile

from export_manual_detection_dataset import BASE,read_json,write_json
from build_center_candidate_manifest import contains,validate_box
from make_centered_review_pack import discover_tifs,read_geotransform,map_source_records,_record_for_tile
from check_tile_connectivity import tiles_of_bbox
import manual_box_review as editor
from fetch_v2_train_tiles import download_zip,extract_verified,existing_tif,MANIFEST,segments

PARENT=BASE/'data/processed/manual_yolo_20261008_v2'
PACK=BASE/'data/processed/cross_tile_neighbor_review_20261008'
EXTRA=BASE/'data/raw/ctx_tiles'


@lru_cache(maxsize=2)
def native_pixels(path,mtime_ns):
    # Retired tiles have lossless compressed native context TIFFs. They retain
    # every source pixel but cannot be memory mapped like the raw CTX mosaics.
    try:return tifffile.memmap(path)
    except ValueError:return tifffile.imread(path)


def compose(entry,max_side=4096):
    """Sample original pixels onto one documented mosaic grid; track coverage."""
    width,height=entry['size_px'];gt=entry['geotransform'];stride=max(1,math.ceil(max(width,height)/max_side))
    xx=gt[0]+np.arange(0,width,stride)*gt[1]
    yy=gt[3]+np.arange(0,height,stride)*gt[5]
    image=np.zeros((len(yy),len(xx)),dtype=np.uint8);coverage=np.zeros(image.shape,dtype=bool)
    for src in entry['sources']:
        path=Path(src['path'])
        if path.stat().st_size!=src['bytes'] or path.stat().st_mtime_ns!=src['mtime_ns']:
            raise ValueError('Mosaic source changed: '+str(path))
        t=src['geotransform'];w,h=src['size_px']
        sx=np.rint((xx-t[0])/t[1]).astype(np.int64);sy=np.rint((yy-t[3])/t[5]).astype(np.int64)
        ix=np.flatnonzero((sx>=0)&(sx<w));iy=np.flatnonzero((sy>=0)&(sy<h))
        if not len(ix) or not len(iy):continue
        region=np.ix_(iy,ix);mask=~coverage[region]
        if not mask.any():continue
        arr=native_pixels(str(path),src['mtime_ns'])
        pixels=np.asarray(arr[np.ix_(sy[iy],sx[ix])])
        old=image[region]
        old[mask]=pixels[mask];image[region]=old;coverage[region]=True
        del arr,pixels
        if coverage.all():break
    return image,coverage,stride


def load_sources(parent):
    plan=read_json(parent/'export_plan.json');idx=plan['inputs']
    with shapefile.Reader(shp=idx['shp']['snapshot'],shx=idx['shx']['snapshot'],dbf=idx['dbf']['snapshot']) as sf:
        sources=map_source_records(sf.shapeRecords())
    return plan,sources


def mosaic_html():
    html=editor.HTML.read_text(encoding='utf-8')
    html=html.replace('<title>滑坡人工修框</title>','<title>跨图幅邻近滑坡审核</title>')
    html=html.replace('<p><strong>背景负样本</strong></p>','<section hidden><p><strong>背景负样本</strong></p>',1).replace('<hr><p><strong>滑坡正样本</strong>','</section><hr><p><strong>滑坡正样本</strong>',1)
    html=html.replace('<option value="background">背景负样本</option>','<option value="background" disabled>背景负样本</option>')
    html=html.replace('<h1>滑坡人工修框</h1>','<h1>跨图幅邻近滑坡审核</h1>')
    html=html.replace('绿色框为滑坡，蓝色框为你确认没有滑坡的背景区域。红色／黄色是原始参考，可关闭参考线。','绿色框为你修订的当前滑坡；红色为当前目标原标注，黄色为相邻目标参考。可关闭参考线。')
    html=html.replace('<label>视野 <select id="factor"><option value="1">原审核范围</option><option value="4" selected>扩大至 4 倍</option><option value="8">扩大至 8 倍</option><option value="16">扩大至 16 倍</option></select></label>',
                      '<label hidden>视野 <select id="factor"><option value="4" selected>完整拼接视野</option></select></label>')
    html=html.replace('如果底部还在画面外，先扩大视野再画。若缺失原图幅外的像素，请暂扣，不能凭空补框。','已拼接当前滑坡涉及的相邻图幅。请核对完整滑坡的顶部和底部，再画框；该页使用完整拼接视野。新下载图幅完成后，刷新页面可查看新增目标。')
    html=html.replace('补全原标注遗漏的滑坡底部','拼接后核对完整滑坡范围')
    html=html.replace('原始多边形、旧训练标签和审核图片保留。','跨图幅修订单独保存，原已确认框和排除决定保留。')
    return html


def build(parent=PARENT,pack=PACK,extra=EXTRA):
    pack.mkdir(parents=True,exist_ok=True);(pack/'previews').mkdir(exist_ok=True)
    plan,sources=load_sources(parent);neighbors=read_json(parent/'held_neighbor_followup.json')['neighbors']
    targets=sorted({r['neighbor_source_id'] for r in neighbors if r['category']=='cross_tile_or_incomplete_neighbor'},
        key=lambda sid:(-sum(len(r['affected_confirmed_source_ids']) for r in neighbors if r['neighbor_source_id']==sid),sid))
    tifs=discover_tifs(extra);tifs.update(discover_tifs(BASE/'data/raw/ctx_tiles'))
    old=read_json(pack/'mosaic_review_manifest.json') if (pack/'mosaic_review_manifest.json').exists() else {'entries':[]}
    old_entries={r['source_id']:r for r in old['entries']};ready=[];pending=[]
    protected=set(plan['protected_tiles'])
    for sid in targets:
        required=tiles_of_bbox(*sources[sid].shape.bbox)
        affected=[dict(tile=r['tile'],source_id=a) for r in neighbors if r['neighbor_source_id']==sid for a in r['affected_confirmed_source_ids']]
        if required&protected:
            pending.append(dict(source_id=sid,reason='protected_eval_target',tiles=sorted(required),affected=affected));continue
        missing=required-set(tifs)
        if missing:
            pending.append(dict(source_id=sid,reason='missing_native_tiles',tiles=sorted(required),missing_tiles=sorted(missing),affected=affected));continue
        if sid in old_entries:
            ready.append(old_entries[sid]);continue
        anchor=next((r['tile'] for r in affected if r['tile'] in required),sorted(required)[0])
        gt,_,_=read_geotransform(tifs[anchor]);srcs=[];limits=[]
        for tile in sorted(required):
            path=tifs[tile];g,w,h=read_geotransform(path)
            if g[1]<=0 or g[5]>=0 or abs(g[1]/gt[1]-1)>.01 or abs(g[5]/gt[5]-1)>.01:
                raise ValueError('Unexpected mosaic pixel resolution')
            srcs.append(dict(tile=tile,path=str(path.resolve()),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,geotransform=list(g),size_px=[w,h]))
            limits.append([(g[0]-gt[0])/gt[1],(g[3]-gt[3])/gt[5],(g[0]+w*g[1]-gt[0])/gt[1],(g[3]+h*g[5]-gt[3])/gt[5]])
        record=_record_for_tile(sid,sources[sid],gt,(-180,-90,180,90));b=record['aabb_px']
        dx=max(256,(b[2]-b[0])*.25);dy=max(256,(b[3]-b[1])*.25)
        x0=max(math.ceil(min(a[0] for a in limits)),math.floor(b[0]-dx));y0=max(math.ceil(min(a[1] for a in limits)),math.floor(b[1]-dy))
        x1=min(math.floor(max(a[2] for a in limits)),math.ceil(b[2]+dx));y1=min(math.floor(max(a[3] for a in limits)),math.ceil(b[3]+dy))
        if not contains([x0,y0,x1,y1],b):raise ValueError('Complete neighbor does not fit mosaic sources')
        mosaic_gt=[gt[0]+x0*gt[1],gt[1],0,gt[3]+y0*gt[5],0,gt[5]]
        own_box=[b[0]-x0,b[1]-y0,b[2]-x0,b[3]-y0]
        entry=dict(tile=f'MOSAIC_{sid:05d}',source_id=sid,anchor_tile=anchor,geotransform=mosaic_gt,
            size_px=[x1-x0,y1-y0],sources=srcs,required_tiles=sorted(required),original_box=own_box,affected=affected,
            resampling='nearest on anchor native grid; source-grid rounding <=0.5 pixel')
        image,coverage,stride=compose(entry)
        px0,py0,px1,py1=own_box
        mask=coverage[max(0,math.floor(py0/stride)):math.ceil(py1/stride)+1,max(0,math.floor(px0/stride)):math.ceil(px1/stride)+1]
        if not mask.size or not mask.all():
            pending.append(dict(source_id=sid,reason='mosaic_coverage_gap',tiles=sorted(required),affected=affected));continue
        preview=pack/'previews'/f'{sid:05d}.png';Image.fromarray(image).save(preview)
        entry.update(preview_path=str(preview.resolve()),preview_stride=stride,preview_size_px=[image.shape[1],image.shape[0]],reference_geometry_complete=True)
        ready.append(entry)
        print(f'MOSAIC_READY {sid} {len(required)} tiles',flush=True)
    value=dict(schema_version=1,purpose='cross-tile neighbors human review, not training labels',parent_dataset=str(parent.resolve()),
        supplemental_ctx_root=str(extra.resolve()),ready_count=len(ready),pending_count=len(pending),entries=ready,pending=pending,
        human_decisions_directory=str((pack/'manual_review').resolve()),updated_epoch=time.time())
    editor.write_json_atomic(pack/'mosaic_review_manifest.json',value)
    # Reuse tested drawing controls, with background controls hidden and disabled.
    temporary=pack/'mosaic_review.html.tmp';temporary.write_text(mosaic_html(),encoding='utf-8');temporary.replace(pack/'mosaic_review.html')
    (pack/'README.md').write_text(f'# 跨图幅邻居审核\n\n当前 {len(ready)} 个完整拼接目标可审核，{len(pending)} 个待补图幅或检查。\n审核地址：http://127.0.0.1:8770/\n原背景网站8769保持运行。\n补充图幅缓存：{extra}\n绿色人工框写入 manual_review/cross_tile_manual_boxes.reviewed.csv，附原生网格坐标及地理变换。尚未人工确认的拼接多边形不会自动成为训练标签。\n',encoding='utf-8')
    return value


class MosaicReview:
    def __init__(self,pack=PACK):
        self.pack=pack;self.folder=pack/'manual_review';self.folder.mkdir(exist_ok=True)
        self.corrections=self.folder/'cross_tile_manual_boxes.reviewed.csv';self.lock=threading.RLock()
        parent=Path(read_json(pack/'mosaic_review_manifest.json')['parent_dataset']);_,self.sources=load_sources(parent)
        self.fields=editor.OVERRIDE_FIELDS+['geotransform','size_px','required_tiles','world_box_m']
    def manifest(self):return read_json(self.pack/'mosaic_review_manifest.json')
    def saved(self):
        if not self.corrections.exists():return {}
        with self.corrections.open(encoding='utf-8-sig') as f:return {f"{r['tile']}:{r['source_id']}":r for r in csv.DictReader(f)}
    def batch_info(self):
        excluded=self.exclusions();saved={k:v for k,v in self.saved().items() if k not in excluded}
        return dict(items=[dict(tile=r['tile'],source_id=r['source_id']) for r in self.manifest()['entries']],saved=saved,excluded=excluded,backgrounds={},background_only=False,batch_number=None,groups=[])
    def select(self,tile):
        entry=next((r for r in self.manifest()['entries'] if r['tile']==tile),None)
        if entry is None:raise ValueError('未知拼接目标')
        return MosaicTarget(self,entry)
    def exclusions(self):
        p=self.folder/'cross_tile_exclusions.json';return read_json(p) if p.exists() else {}
    def save(self,payload):
        target=self.select(payload['tile']);e=target.entry;sid=int(payload['source_id']);w,h=e['size_px']
        editor.validate_manual(payload,{sid:{'aabb_px':e['original_box']}},w,h)
        if sid!=e['source_id']:raise ValueError('拼接目标编号不匹配')
        box=validate_box(payload['box']);_,mask,stride=target.pixels()
        xx0,yy0,xx1,yy1=box;region=mask[math.floor(yy0/stride):math.ceil(yy1/stride),math.floor(xx0/stride):math.ceil(xx1/stride)]
        if not region.size or not region.all():raise ValueError('新框包含缺失像素，请暂扣')
        gt=e['geotransform'];world=[gt[0]+box[0]*gt[1],gt[3]+box[1]*gt[5],gt[0]+box[2]*gt[1],gt[3]+box[3]*gt[5]]
        row=dict(tile=e['tile'],source_id=str(sid),**dict(zip(['x0','y0','x1','y1'],box)),label_scope=payload['label_scope'],reviewer=payload['reviewer'],confirmed='true',reason=payload['reason'],geotransform=json.dumps(gt),size_px=json.dumps(e['size_px']),required_tiles=json.dumps(e['required_tiles']),world_box_m=json.dumps(world))
        with self.lock:
            rows=list(self.saved().values());key=f"{e['tile']}:{sid}";rows=[r for r in rows if f"{r['tile']}:{r['source_id']}"!=key]+[row]
            temp=self.corrections.with_suffix('.tmp')
            with temp.open('w',encoding='utf-8-sig',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=self.fields);wr.writeheader();wr.writerows(rows)
            temp.replace(self.corrections)
            exclusions=self.exclusions()
            if key in exclusions:del exclusions[key];editor.write_json_atomic(self.folder/'cross_tile_exclusions.json',exclusions)
            with (self.folder/'cross_tile_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(action='save',record=row,saved_epoch=time.time()),ensure_ascii=False)+'\n')
        return row
    def exclude(self,payload):
        e=self.select(payload['tile']).entry
        if int(payload['source_id'])!=e['source_id'] or not str(payload.get('reviewer','')).strip() or not str(payload.get('reason','')).strip():raise ValueError('请核对编号并填写审核依据')
        row=dict(tile=e['tile'],source_id=e['source_id'],training_use='exclude_positive',negative_eligible=False,reviewer=payload['reviewer'],reason=payload['reason'],geotransform=e['geotransform'],exclusion_box_px=e['original_box'],saved_epoch=time.time())
        with self.lock:
            d=self.exclusions();d[f"{e['tile']}:{e['source_id']}"]=row;editor.write_json_atomic(self.folder/'cross_tile_exclusions.json',d)
            key=f"{e['tile']}:{e['source_id']}";saved=self.saved()
            if key in saved:
                rows=[r for k,r in saved.items() if k!=key];temp=self.corrections.with_suffix('.tmp')
                with temp.open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=self.fields);writer.writeheader();writer.writerows(rows)
                temp.replace(self.corrections)
            with (self.folder/'cross_tile_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(action='exclude',record=row,saved_epoch=time.time()),ensure_ascii=False)+'\n')
        return row
    def save_background(self,p):raise ValueError('此页面仅审核拼接滑坡；背景审核请使用8769页面')
    def remove_background(self,p):raise ValueError('此页面不保存背景')


class MosaicTarget:
    def __init__(self,review,entry):self.review,self.entry=review,entry
    def pixels(self):return mosaic_pixels(json.dumps(self.entry,sort_keys=True))
    def view(self,sid,factor):
        if sid!=self.entry['source_id'] or factor not in {1,4,8,16}:raise ValueError('未知目标或视野')
        e=self.entry;image,coverage,stride=self.pixels();polys=[]
        for source_id,r in self.review.sources.items():
            a=_record_for_tile(source_id,r,tuple(e['geotransform']),geo_bounds(e))
            if a:polys.append(dict(source_id=source_id,polygon_px=a['polygon_px']))
        buffer=io.BytesIO();Image.fromarray(image).save(buffer,format='PNG');w,h=e['size_px']
        return dict(tile=e['tile'],source_id=sid,window=[0,0,w,h],stride=stride,image_width=image.shape[1],image_height=image.shape[0],original_box=e['original_box'],polygons=polys,tile_width=w,tile_height=h,geometry_complete=True,cached_context=False,available_window_px=[0,0,w,h],required_tiles=e['required_tiles'],resampling=e['resampling']),buffer.getvalue()


def geo_bounds(e):
    gt=e['geotransform'];w,h=e['size_px'];k=3396190*math.pi/180
    return (gt[0]/k,(gt[3]+h*gt[5])/k,(gt[0]+w*gt[1])/k,gt[3]/k)


@lru_cache(maxsize=2)
def mosaic_pixels(entry_json):return compose(json.loads(entry_json))


def download_missing_locked(parent,pack,extra):
    value=build(parent,pack,extra)
    remaining=sorted({t for r in value['pending'] for t in r.get('missing_tiles',[])})
    report=pack/'supplemental_download_status.json'
    previous=read_json(report) if report.exists() else {}
    wanted=sorted(set(previous.get('wanted',[]))|set(remaining))
    if not wanted:return
    catalog={r['tile']:r for r in read_json(MANIFEST)};extra.mkdir(parents=True,exist_ok=True)
    # Budget retained native tiles and the peak ZIP assembly/extraction scratch.
    # Valid resumable prefixes already occupy disk and are released on success.
    growth=0;peak=0
    for tile in remaining:
        row=catalog[tile];total=row['size_bytes'];credit=0
        zip_path=extra/row['zip']
        if zip_path.exists() and zip_path.stat().st_size<=total:credit+=zip_path.stat().st_size
        for i,(a,b) in enumerate(segments(total,4)):
            part=extra/'_v2_download_parts'/tile/f'part_{i:02d}_{a}_{b}.bin'
            if part.exists() and part.stat().st_size<=b-a+1:credit+=part.stat().st_size
        native=2_250_000_000
        peak=max(peak,growth+max(2*total,total+native)-credit)
        growth+=native-credit
    required=max(0,peak)+(3<<30) if remaining else 0
    if shutil.disk_usage(extra).free < required:
        raise ValueError(f'Insufficient free space for {len(remaining)} supplemental tiles: need {required} bytes')
    verified_tiles={}
    for tile in set(wanted)-set(remaining):
        verified=existing_tif(extra,tile)
        if verified is None:raise ValueError('Previously completed source is missing: '+tile)
        verified_tiles[tile]=dict(status='verified',**verified)
    state=dict(status='running',wanted=wanted,tiles=verified_tiles,remaining_at_resume=remaining,
               cache_root=str(extra.resolve()),required_additional_bytes=required,free_bytes_at_resume=shutil.disk_usage(extra).free,purpose='specific cross-tile review mosaics; no frozen split changes')
    editor.write_json_atomic(report,state)
    for tile in remaining:
        row=catalog[tile];expected=f'MurrayLab_GlobalCTXMosaic_V01_{tile}.zip'
        if row['zip']!=expected or row['url']!=f'https://murray-lab.caltech.edu/CTX/V01/tiles/{expected}':raise ValueError('Unexpected download URL')
        try:
            state['tiles'][tile]={'status':'downloading'};editor.write_json_atomic(report,state)
            verified=existing_tif(extra,tile)
            if verified is None:verified=extract_verified(row,download_zip(row,extra,4),extra)
            state['tiles'][tile]=dict(status='verified',**verified);editor.write_json_atomic(report,state)
            print('VERIFIED '+tile,flush=True);build(parent,pack,extra)
        except Exception as exc:
            state['tiles'][tile]={'status':'failed','error':str(exc)};editor.write_json_atomic(report,state)
    state['status']='complete' if all(r['status']=='verified' for r in state['tiles'].values()) else 'failed';editor.write_json_atomic(report,state)


def download_missing(parent,pack,extra,wait_for_lock=False):
    import msvcrt
    extra.mkdir(parents=True,exist_ok=True)
    with (extra/'_v2_restore.lock').open('a+b') as handle:
        if not handle.tell():handle.write(b'0');handle.flush()
        deadline=time.monotonic()+3600;announced=False
        while True:
            try:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError as exc:
                if not wait_for_lock or exc.errno not in {13,11} or time.monotonic()>=deadline:raise
                if not announced:print('WAITING_FOR_ACTIVE_DOWNLOAD; one resumable retry pass follows',flush=True);announced=True
                time.sleep(10)
        try:download_missing_locked(parent,pack,extra)
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,default=PARENT);ap.add_argument('--pack',type=Path,default=PACK);ap.add_argument('--extra-ctx',type=Path,default=EXTRA);ap.add_argument('--serve',action='store_true');ap.add_argument('--download',action='store_true');ap.add_argument('--wait-for-lock',action='store_true');ap.add_argument('--port',type=int,default=8770);args=ap.parse_args()
    if args.download:download_missing(args.parent,args.pack,args.extra_ctx,args.wait_for_lock)
    elif args.serve:
        editor.HTML=args.pack/'mosaic_review.html';server=ThreadingHTTPServer(('127.0.0.1',args.port),editor.handler_for(MosaicReview(args.pack)));print(f'CROSS_TILE_REVIEW http://127.0.0.1:{args.port}/',flush=True);server.serve_forever()
    else:
        v=build(args.parent,args.pack,args.extra_ctx);print(json.dumps(dict(ready=v['ready_count'],pending=v['pending_count']),ensure_ascii=False))
