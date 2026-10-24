"""Versioned SQLite import and read-only catalog adapter."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from api.schemas import Catalog

SCHEMA = Path(__file__).parent / 'db/schema.sql'


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def build_database(path: Path, payload: dict | list, *, raw=False,
                   source_title='老师提供的火星滑坡编目', source_index: list | None = None,
                   image_index: list | None = None, asset_root: Path | None = None) -> dict:
    """Create a new database; raw imports stay staged until coordinates are verified."""
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('Database already exists; write a new version')
    if raw:
        if not isinstance(payload,list) or not payload:raise ValueError('Raw input must be a nonempty list')
        version='inventory-staging-20261024';mode='catalog';state='staging'
        coordinates={'body':'Mars','longitude':'east-positive, -180..180','latitude':'unverified'}
        index={int(r['source_id']):r for r in (source_index or [])}
        records=[]
        for r in payload:
            sid=r['source_id']
            if isinstance(sid,bool) or not isinstance(sid,int) or sid<0:raise ValueError('Invalid source_id')
            records.append(dict(id=r['id'],sourceId=sid,name=r.get('name') or f'滑坡 {sid}',type=r['type'],
                longitude=r['lon'],latitude=r['lat'],areaKm2=r.get('area'),summary=r.get('desc') or '',
                imageUrl=None,source={'title':source_title,'url':'https://doi.org/10.13203/j.whugis20250300'},isDemo=False,
                rawType=index.get(sid,{}).get('raw_type',r['type']),perimeter=r.get('perimeter'),original=r))
    else:
        catalog=Catalog.model_validate(payload)
        version=catalog.version;mode=catalog.mode;state='ready';coordinates=catalog.coordinateSystem.model_dump()
        records=[dict(r.model_dump(),rawType=r.type,perimeter=None,original=r.model_dump()) for r in catalog.records]
    ids=set();source_ids=set()
    for r in records:
        if not isinstance(r['id'],str) or not r['id'] or r['id'] in ids:raise ValueError('Duplicate or empty id')
        ids.add(r['id'])
        sid=r['sourceId']
        if sid is not None:
            if sid in source_ids:raise ValueError('Duplicate source_id')
            source_ids.add(sid)
        if r['type'] not in ('SL','SP','FL','CL'):raise ValueError('Unknown type')
        for k,limit in [('longitude',180),('latitude',90)]:
            v=r[k]
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or abs(v)>limit:raise ValueError('Invalid coordinates')
        if r['areaKm2'] is not None and (isinstance(r['areaKm2'],bool) or not isinstance(r['areaKm2'],(int,float))
                or not math.isfinite(r['areaKm2']) or r['areaKm2']<0):raise ValueError('Invalid area')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.building')
    if temporary.exists():raise FileExistsError('Prior build exists; inspect before retrying')
    db=sqlite3.connect(temporary)
    try:
        db.executescript(SCHEMA.read_text(encoding='utf-8'))
        metadata={'version':version,'mode':mode,'release_state':state,'coordinateSystem':coordinates,
            'created_utc':datetime.now(timezone.utc).isoformat(),'schema_version':1}
        with db:
            db.executemany('INSERT INTO metadata VALUES (?,?)',[(k,_json(v)) for k,v in metadata.items()])
            source_refs={}
            for r in records:
                key=(r['source']['title'],r['source']['url'])
                if key not in source_refs:
                    ref=f'source-{len(source_refs)+1}';source_refs[key]=ref
                    db.execute('INSERT INTO sources(id,title,url) VALUES (?,?,?)',(ref,*key))
                db.execute('INSERT INTO landslides VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(
                    r['id'],r['sourceId'],r['name'],r['type'],r['rawType'],r['longitude'],r['latitude'],r['areaKm2'],
                    r['perimeter'],r['summary'],r['imageUrl'],source_refs[key],int(r['isDemo']),_json(r['original'])))
            id_by_source={r['sourceId']:r['id'] for r in records if r['sourceId'] is not None}
            for a in image_index or []:
                target=id_by_source.get(int(a['source_id']))
                if target not in ids:raise ValueError('Image index points outside catalog')
                relative=Path(a['image_path']);asset_path=(asset_root/relative).resolve() if asset_root else relative
                db.execute('INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)',(
                    f"train224-{a['source_id']}",target,'image',str(asset_path),'image/png',a.get('image_sha256'),224,224,
                    _json({'source_tiles':a.get('source_tiles',[]),'purpose':'preview','original_index_path':a['image_path']})))
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        count=db.execute('SELECT count(*) FROM landslides').fetchone()[0]
    except BaseException:
        db.close();temporary.unlink(missing_ok=True);raise
    db.close();temporary.rename(path)
    return dict(path=str(path),records=count,release_state=state,mode=mode)


def load_sqlite_catalog(path: Path) -> Catalog:
    path=Path(path).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        if db.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise ValueError('Unsupported database schema version')
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Database has invalid foreign-key references')
        metadata={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}
        if metadata.get('release_state')!='ready':
            raise ValueError('Catalog is staged: coordinate convention has not been approved')
        rows=db.execute('''SELECT l.*,s.title AS source_title,s.url AS source_url
            FROM landslides l JOIN sources s ON s.id=l.source_ref ORDER BY l.rowid''').fetchall()
        records=[dict(id=r['id'],sourceId=r['source_id'],name=r['name'],type=r['type'],longitude=r['longitude'],
            latitude=r['latitude'],areaKm2=r['area_km2'],summary=r['summary'],imageUrl=r['image_url'],
            source={'title':r['source_title'],'url':r['source_url']},isDemo=bool(r['is_demo'])) for r in rows]
        return Catalog.model_validate(dict(version=metadata['version'],mode=metadata['mode'],
            coordinateSystem=metadata['coordinateSystem'],records=records))
