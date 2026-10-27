"""Guard the tracked publication boundary; no private data content is printed."""
import json
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
    demo=json.loads((ROOT/'examples/catalog.demo.json').read_text(encoding='utf-8'))
    assert demo['mode']=='demo' and all(r['isDemo'] is True for r in demo['records'])
    print(f'Publication boundary passed: {len(tracked)} tracked files;8 real metadata samples; no full data/model assets')


if __name__=='__main__':main()
