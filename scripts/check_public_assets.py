"""Guard the tracked publication boundary; no private data content is printed."""
import json
import hashlib
import struct
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_PREFIXES=('runs/','weights/','output/','tmp/','data/raw/','data/processed/','data/manifest/')
PRIVATE_SUFFIXES={'.shp','.shx','.dbf','.sbn','.sbx','.prj','.tif','.tiff','.safetensors',
    '.pt','.pth','.onnx','.npz','.npy','.sqlite3','.db','.joblib','.zip','.tar','.gz','.csv','.geojson'}
ALLOWED_PLACEHOLDERS={'data/raw/sample/.gitkeep'}
PUBLIC_SOURCE_IDS={482, 483, 493, 528, 529, 533, 597, 505}


def check_subset(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    assert set(value)=={'version','status','coordinateStatus','records'},'Unexpected real-sample fields'
    assert value['status']=='reference_subset_not_runtime_catalog'
    rows=value['records'];assert len(rows)==8,'Only the explicitly selected8 records may be published'
    assert {r['type'] for r in rows}=={'SL','SP','FL','CL'}
    assert all(sum(r['type']==c for r in rows)==2 for c in ['SL','SP','FL','CL'])
    assert {r['sourceId'] for r in rows}==PUBLIC_SOURCE_IDS,'Public sample IDs changed; review the release scope explicitly'
    fields={'id','sourceId','name','type','longitude','latitude','areaKm2','summary','imageUrl','source','isDemo'}
    for r in rows:
        assert set(r)==fields and r['isDemo'] is False and r['imageUrl'] is None
        assert set(r['source'])=={'title','url'}
    assert path.stat().st_size<32768,'Unexpected sample payload size'


def real_source_ids(value):
    found=set()
    if isinstance(value,dict):
        sid=value.get('sourceId',value.get('source_id'))
        if value.get('type') in ('SL','SP','FL','CL','RL','DF') and isinstance(sid,int) and not isinstance(sid,bool):found.add(sid)
        for child in value.values():found.update(real_source_ids(child))
    elif isinstance(value,list):
        for child in value:found.update(real_source_ids(child))
    return found


def check_images(tracked):
    """Allow only the eight selected, metadata-free 224px training previews."""
    manifest=json.loads((ROOT/'examples/recognition.samples.json').read_text(encoding='utf-8'))
    rows=manifest['records']
    assert len(rows)==8 and {r['sourceId'] for r in rows}==PUBLIC_SOURCE_IDS
    metadata={r['sourceId']:r for r in json.loads((ROOT/'examples/catalog.real-subset.json').read_text(encoding='utf-8'))['records']}
    allowed=set()
    for r in rows:
        assert set(r)=={'id','sourceId','type','imageUrl','width','height','roi','sha256','usage','source'}
        assert r['id']==metadata[r['sourceId']]['id'] and r['type']==metadata[r['sourceId']]['type']
        assert r['source']==metadata[r['sourceId']]['source']
        assert r['usage']=='training_example_not_evaluation'
        assert r['width']==r['height']==224 and r['imageUrl']=='/samples/'+r['id']+'.png'
        roi=r['roi'];assert len(roi)==4 and all(isinstance(v,(int,float)) for v in roi)
        assert 0<=roi[0]<roi[2]<=224 and 0<=roi[1]<roi[3]<=224
        relative='mobile/public'+r['imageUrl'];allowed.add(relative)
        raw=(ROOT/relative).read_bytes()
        assert len(raw)<100000 and hashlib.sha256(raw).hexdigest()==r['sha256']
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        offset=8;chunks=[]
        while offset<len(raw):
            size=struct.unpack('>I',raw[offset:offset+4])[0]
            kind=raw[offset+4:offset+8];chunks.append(kind)
            if kind==b'IHDR':assert struct.unpack('>II',raw[offset+8:offset+16])==(224,224)
            offset+=size+12
        assert offset==len(raw) and chunks[0]==b'IHDR' and chunks[-1]==b'IEND'
        assert set(chunks)<={b'IHDR',b'IDAT',b'IEND'},'Unexpected image metadata'
    published={p for p in tracked if p.startswith('mobile/public/samples/')}
    assert published==allowed,'Unapproved or missing sample images'
    images={p for p in tracked if Path(p).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.gif','.bmp'}}
    assert images==allowed|{'mobile/public/textures/mars.jpg'},'Unexpected image outside the approved release set'


def main():
    result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,check=True,capture_output=True)
    tracked=[p for p in result.stdout.decode('utf-8').split('\0') if p]
    blocked=[]
    for relative in tracked:
        if relative in ALLOWED_PLACEHOLDERS:continue
        path=ROOT/relative
        if relative.startswith(PRIVATE_PREFIXES) or path.suffix.lower() in PRIVATE_SUFFIXES:
            blocked.append(relative)
        elif path.is_file() and path.stat().st_size>2*1024*1024:
            blocked.append(relative)
        elif path.name.endswith(('.sqlite3.building','.sqlite3-wal','.sqlite3-shm','.sqlite3-journal')):
            blocked.append(relative)
        elif path.name in ['.env','.env.local','credentials.json','dashscope.key']:
            blocked.append(relative)
        if path.is_file() and path.suffix.lower()=='.json':
            assert real_source_ids(json.loads(path.read_text(encoding='utf-8'))) <= PUBLIC_SOURCE_IDS, 'Real records outside the approved subset: '+relative
    if blocked:raise SystemExit('Private/local assets are tracked: '+', '.join(blocked))
    check_subset(ROOT/'examples/catalog.real-subset.json')
    check_images(tracked)
    demo=json.loads((ROOT/'examples/catalog.demo.json').read_text(encoding='utf-8'))
    assert demo['mode']=='demo' and all(r['isDemo'] is True for r in demo['records'])
    print(f'Publication boundary passed: {len(tracked)} tracked files;8 metadata samples and8 approved 224px previews; no full data/model assets')


if __name__=='__main__':main()
