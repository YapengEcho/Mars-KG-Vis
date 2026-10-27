"""Review whole held-out images in a separate, versioned sidecar. No training writes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import random
import threading
from urllib.parse import parse_qs, urlparse
from PIL import Image
from audit_yolo_splits import parse_patch_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'runs/ablations/eval_label_review_20261009'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)

def original_boxes(path, width, height):
    boxes = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split())
        if cls != 0:
            raise ValueError('Expected the frozen single-class evaluation labels')
        box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        boxes.append([max(0., min(float(width if i % 2 == 0 else height), v)) for i, v in enumerate(box)])
    return boxes

def prepare(out=DEFAULT, seed=20261009):
    out = Path(out)
    if (out/'manifest.json').exists():
        raise FileExistsError('Review selection is already frozen; use serve to resume')
    dataset = ROOT/'data/processed/manual_yolo_20261009_v8/manual'
    old_manifest = ROOT/'data/manifest/yolo_tile_flval_round3_multilabel_stage1.json'
    catalog = read(old_manifest)
    frozen = {r['path']: r['sha256'] for r in catalog['files']}
    leakage = ROOT/'runs/ablations/train_eval_leakage_20261009/audit.json'
    audit = read(leakage)
    assert audit['status'] == 'complete_read_only_leakage_audit' and not audit['leakage_detected']
    groups, hashes, duplicates = defaultdict(list), {}, []
    images = sorted((dataset/'images/test').glob('*.png'))
    assert len(images) == 497
    train_tiles = set()
    for folder in [dataset/'images/train', ROOT/'data/processed/yolo_tile_flval_round3_multilabel_stage1/images/train']:
        for image in folder.glob('*.png'):
            # New manual crop filenames differ; their tile prefix is the same.
            import re
            match = re.match(r'(E-?\d{3}_N-?\d{2})_', image.stem)
            if match:
                train_tiles.add(match[1])
    for image in images:
        label = dataset/'labels/test'/f'{image.stem}.txt'
        image_hash, label_hash = sha(image), sha(label)
        assert image_hash == frozen[f'images/test/{image.name}']
        assert label_hash == frozen[f'labels/test/{label.name}']
        meta = parse_patch_name(image.stem)
        assert meta['tile'] not in train_tiles
        if image_hash in hashes:
            canonical = hashes[image_hash]
            with Image.open(image) as im:
                width, height = im.size
            boxes = original_boxes(label, width, height)
            assert sorted(boxes) == sorted(canonical['original_boxes']), 'Duplicate pixels have conflicting geometry'
            duplicates.append(dict(stem=image.stem, canonical=canonical['stem'],
                                   label_sha256=label_hash, geometry_equal=True,
                                   label_row_order_differs=label_hash!=canonical['label_sha256']))
            continue
        with Image.open(image) as im:
            width, height = im.size
        row = dict(stem=image.stem, tile=meta['tile'], split='test', image=str(image.resolve()),
                   label=str(label.resolve()), image_sha256=image_hash, label_sha256=label_hash,
                   width=width, height=height, original_boxes=original_boxes(label, width, height),
                   source_window=meta)
        hashes[image_hash] = row
        groups[meta['tile']].append(row)
    rng = random.Random(seed)
    for tile in sorted(groups):
        rng.shuffle(groups[tile])
    # Balanced round-robin across all four test tiles. No prediction/score selection.
    items = []
    for index in range(max(map(len, groups.values()))):
        for tile in sorted(groups):
            if index < len(groups[tile]):
                items.append(groups[tile][index])
    for index, item in enumerate(items):
        item['batch'] = index//100+1
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
        role='human relabeling diagnostic of the existing held-out test; never training',
        fresh_blind_test=False, selection='SHA-deduplicated; seeded shuffle per tile then round-robin; no predictions',
        batch_size=100, frozen_catalog_sha256=sha(old_manifest), leakage_audit_sha256=sha(leakage),
        original_test_images=len(images), unique_review_images=len(items), duplicates=duplicates, items=items,
        policies=dict(box_scope='Whole identifiable landslide visible in the fixed image, including head and runout; clip at image boundary.',
            truncation='Retain identifiable partial objects with an edge flag. Do not infer unseen pixels.',
            uncertain='Mark the entire image unscorable if an apparent landslide or boundary cannot be adjudicated; never turn it into confirmed background.',
            empty='Require explicit whole-image no-landslide confirmation.',
            evaluation='Freeze human labels before showing predictions. Reuse all six existing best checkpoints and validation-locked thresholds; compare old vs human labels on the exact same accepted images. No retuning.',
            limitations='Existing test images have been inspected historically. This is an annotation diagnostic, not a new blind benchmark. Stratified first batch is not the full-test aggregate.'))
    out.mkdir(parents=True, exist_ok=True)
    atomic(out/'manifest.json', manifest)
    atomic(out/'decisions.json', dict(schema_version=1, manifest_sha256=sha(out/'manifest.json'), records={}))
    (out/'README.md').write_text('人工评估标注诊断\n\n保持原测试图片像素及裁剪；不进入训练。首批100张，按四个图幅均衡抽取，包含原有空标签图片。\n\n逐张检查整个画面，框出所有可识别滑坡（顶部、滑体、底部），裁片边缘的滑坡仅框可见部分并保留。原框可按需显示或载入草稿；模型预测不显示。\n\n看不清的整张图片暂扣，不能当背景；只有确认全图无滑坡时才保存空标签。完成一批后，冻结人工标签，固定现有权重和阈值，在完全相同图片上分别用旧框与人工框评分。不用于调参，也不覆盖原评估集。\n', encoding='utf-8')
    return dict(images=len(items), duplicate_pixels_removed=len(duplicates), first_batch=len(items[:100]),
                first_batch_tiles=dict(Counter(r['tile'] for r in items[:100])),
                first_batch_old_empty=sum(not r['original_boxes'] for r in items[:100]))

class Review:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = read(self.folder/'manifest.json')
        self.hash = sha(self.folder/'manifest.json')
        self.items = {row['stem']: row for row in self.manifest['items']}
        self.lock = threading.Lock()
        self.state = read(self.folder/'decisions.json')
        assert self.state['manifest_sha256'] == self.hash

    def save(self, payload):
        stem = payload.get('stem')
        if stem not in self.items:
            raise ValueError('图片不在审核清单中')
        item = self.items[stem]
        status = payload.get('status')
        if status not in {'accepted', 'unscorable'}:
            raise ValueError('未知审核状态')
        if payload.get('confirmed') is not True:
            raise ValueError('请勾选整张图片审核确认')
        reviewer, reason = str(payload.get('reviewer', '')).strip(), str(payload.get('reason', '')).strip()
        if not reviewer or not reason or len(reviewer)>100 or len(reason)>2000:
            raise ValueError('请填写审核人和判断依据')
        boxes = payload.get('boxes')
        if not isinstance(boxes, list) or len(boxes)>300:
            raise ValueError('无效目标列表')
        clean = []
        for row in boxes:
            if not isinstance(row, list) or len(row)!=4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row):
                raise ValueError('框坐标无效')
            a, b, c, d = map(float, row)
            if not 0<=a<c<=item['width'] or not 0<=b<d<=item['height']:
                raise ValueError('框必须在图片内且面积大于零')
            clean.append([a, b, c, d])
        if status == 'accepted' and not clean and payload.get('empty_confirmed') is not True:
            raise ValueError('空标签需要单独确认整张图没有滑坡')
        if sha(item['image']) != item['image_sha256'] or sha(item['label']) != item['label_sha256']:
            raise ValueError('原始图片或标签已经改变，停止保存')
        with self.lock:
            if sha(self.folder/'manifest.json') != self.hash:
                raise ValueError('固定清单发生变化')
            self.state = read(self.folder/'decisions.json')
            previous = self.state['records'].get(stem)
            revision = previous['revision'] if previous else 0
            if payload.get('revision') != revision:
                raise ValueError('该图片已由另一页面保存，请刷新再审核')
            record = dict(stem=stem, image_sha256=item['image_sha256'], status=status, boxes=clean,
                touches_edge=[a<=1 or b<=1 or c>=item['width']-1 or d>=item['height']-1 for a,b,c,d in clean],
                reviewer=reviewer, reason=reason, confirmed=True,
                empty_confirmed=bool(payload.get('empty_confirmed')) if not clean else False,
                revision=revision+1, saved_utc=datetime.now(timezone.utc).isoformat())
            with (self.folder/'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(previous=previous, saved=record), ensure_ascii=False)+'\n')
            self.state['records'][stem] = record
            atomic(self.folder/'decisions.json', self.state)
            return record

def export(folder, batch=1):
    """Immutable accepted-image label snapshot; refuse any incomplete batch."""
    review = Review(folder)
    selected = [r for r in review.manifest['items'] if r['batch']==batch]
    if not selected or any(r['stem'] not in review.state['records'] for r in selected):
        raise ValueError('本批未全部审核，不能冻结评估标签')
    destination = review.folder/f'frozen_batch_{batch:02d}'
    if destination.exists():
        raise FileExistsError('本批已经冻结，不能覆盖')
    rows, held = [], []
    for item in selected:
        record = review.state['records'][item['stem']]
        if record['status']=='unscorable':
            held.append(record)
            continue
        assert sha(item['image'])==item['image_sha256'] and sha(item['label'])==item['label_sha256']
        rows.append(dict(**item, human=record))
    if not rows:
        raise ValueError('本批没有可评分图片')
    destination.mkdir()
    for arm in ['original', 'human']:
        target = destination/arm/'labels'; target.mkdir(parents=True)
        for item in rows:
            boxes = item['original_boxes'] if arm=='original' else item['human']['boxes']
            lines = ['0 {:.9f} {:.9f} {:.9f} {:.9f}'.format((a+c)/2/item['width'], (b+d)/2/item['height'], (c-a)/item['width'], (d-b)/item['height']) for a,b,c,d in boxes]
            (target/f"{item['stem']}.txt").write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')
    result = dict(manifest_sha256=review.hash, decisions_sha256=sha(review.folder/'decisions.json'),
        batch=batch, accepted=rows, held=held, created_utc=datetime.now(timezone.utc).isoformat(),
        fresh_blind_test=False, threshold_policy='Reuse original validation locks; no checkpoint selection or tuning')
    atomic(destination/'frozen.json', result)
    return dict(folder=str(destination), accepted=len(rows), held=len(held))

def serve(folder, port=8773):
    review = Review(folder)
    reference_path = review.folder/'shp_reference.json'
    shp_reference = read(reference_path) if reference_path.exists() else None
    if shp_reference and shp_reference['manifest_sha256'] != review.hash:
        raise ValueError('SHP参考与当前固定清单不一致')
    class Handler(BaseHTTPRequestHandler):
        def response(self, value, status=200, content_type='application/json; charset=utf-8'):
            blob = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(blob)))
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(blob)
        def do_GET(self):
            try:
                url = urlparse(self.path); query = parse_qs(url.query)
                if url.path=='/':
                    return self.response(Path(__file__).with_suffix('.html').read_bytes(), content_type='text/html; charset=utf-8')
                if url.path=='/api/state':
                    return self.response(dict(manifest=review.manifest, records=read(review.folder/'decisions.json')['records']))
                if url.path=='/api/shp':
                    stem=query.get('stem', [''])[0]
                    if stem not in review.items:
                        raise ValueError('图片不在审核清单中')
                    return self.response(dict(available=shp_reference is not None,
                        polygons=shp_reference['items'].get(stem, []) if shp_reference else []))
                if url.path=='/api/image':
                    item = review.items[query.get('stem', [''])[0]]
                    blob = Path(item['image']).read_bytes()
                    if hashlib.sha256(blob).hexdigest()!=item['image_sha256']:
                        raise ValueError('图片像素校验失败')
                    return self.response(blob, content_type='image/png')
                self.response(dict(error='Not found'), 404)
            except (ValueError, KeyError) as error:
                self.response(dict(error=str(error)), 400)
        def do_POST(self):
            try:
                if self.path!='/api/save' or self.headers.get('X-Eval-Review')!='1':
                    raise ValueError('Invalid review request')
                origin = self.headers.get('Origin')
                if origin and origin!=f"http://{self.headers.get('Host')}":
                    raise ValueError('Cross-origin save denied')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0<size<=131072:
                    raise ValueError('Invalid payload size')
                self.response(dict(saved=review.save(json.loads(self.rfile.read(size)))))
            except (ValueError, KeyError, TypeError) as error:
                self.response(dict(error=str(error)), 400)
    print(f'Whole-image evaluation review http://127.0.0.1:{port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'serve', 'export'])
    parser.add_argument('--folder', type=Path, default=DEFAULT)
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    if args.mode=='serve':
        serve(args.folder, args.port)
    else:
        print(json.dumps(prepare(args.folder) if args.mode=='prepare' else export(args.folder, args.batch), ensure_ascii=False))

if __name__=='__main__':
    main()
