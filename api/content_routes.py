"""Read-only atlas, case, citation and bounded resource endpoints."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from api.content_store import content_payload, metadata, read_database
from api.schemas import SlideType

router = APIRouter(prefix='/api', tags=['Content'])


def fail(status, code, message):
    raise HTTPException(status_code=status, detail={'code':code,'message':message})


def database(request):
    path = request.app.state.content_database
    if path is None:
        fail(404,'CONTENT_NOT_CONFIGURED','SQLite content is not configured.')
    return read_database(path)


def envelope(request, **values):
    catalog=request.app.state.catalog
    return dict(dataMode=catalog.mode,version=catalog.version,**values)


def visible_ids(request):
    return {r.id for r in request.app.state.catalog.records}


def source_docs(db, case_id):
    return [dict(r) for r in db.execute('''SELECT d.id,d.title,d.url,d.citation,c.locator
        FROM case_documents c JOIN documents d ON d.id=c.document_id
        WHERE c.case_id=? ORDER BY d.id,c.locator''',(case_id,))]


def case_view(db, row, request):
    slide=next(r for r in request.app.state.catalog.records if r.id==row['landslide_id'])
    image=row['image_asset_id']
    if image is not None:
        owner=db.execute('SELECT landslide_id,kind FROM assets WHERE id=?',(image,)).fetchone()
        if owner is None or owner['landslide_id']!=slide.id or owner['kind']!='image':
            fail(409,'RESOURCE_MISMATCH','Case image does not match its landslide.')
    if image is None:
        media=db.execute("SELECT id FROM assets WHERE landslide_id=? AND kind='image' ORDER BY id LIMIT 1",(slide.id,)).fetchone()
        image=media[0] if media else None
    return dict(id=row['id'],landslideId=slide.id,type=slide.type,title=row['title'],
        description=row['description'],status=row['status'],landslide=slide.model_dump(),
        content=content_payload(db,'case',row['id']),documents=source_docs(db,row['id']),
        imageUrl=f'/api/assets/{image}/content' if image else None,
        geometry=[dict(kind=r['kind'],url=f'/api/landslides/{slide.id}/geometry?kind={r["kind"]}')
                  for r in db.execute('SELECT kind FROM geometry_links WHERE landslide_id=? ORDER BY kind',(slide.id,))])


@router.get('/atlas')
def atlas(request: Request):
    if request.app.state.content_database is None:
        return envelope(request,contentVersion=None,types=[],comparisons=[],limits=[])
    with database(request) as db:
        meta=metadata(db)
        published_cases={r['id'] for r in db.execute("SELECT id,landslide_id FROM cases WHERE status='ready'")
                         if r['landslide_id'] in visible_ids(request)}
        items=[]
        for row in db.execute("SELECT * FROM atlas_types WHERE status='ready' ORDER BY type"):
            detail=content_payload(db,'type',row['type'])
            if detail is not None:
                detail['caseIds']=[cid for cid in detail.get('caseIds',[]) if cid in published_cases]
            items.append(dict(type=row['type'],title=row['title'],description=row['description'],content=detail))
        comparisons=[]
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='atlas_content'").fetchone():
            allowed={i['type'] for i in items}
            comparisons=[json.loads(r[0]) for r in db.execute("SELECT payload_json FROM atlas_content WHERE kind='comparison' ORDER BY id")]
            comparisons=[c for c in comparisons if c['left'] in allowed and c['right'] in allowed]
        return envelope(request,contentVersion=meta.get('atlasContentVersion'),types=items,
                        comparisons=comparisons,limits=meta.get('atlasLimits',[]) if items else [])


@router.get('/atlas/{slide_type}')
def atlas_type(slide_type: SlideType, request: Request):
    overview=atlas(request)
    item=next((i for i in overview['types'] if i['type']==slide_type),None)
    if item is None:fail(404,'NOT_FOUND','Published type was not found.')
    return envelope(request,contentVersion=overview['contentVersion'],item=item)


@router.get('/cases')
def list_cases(request: Request, type: SlideType | None=None,
               page: int=Query(1,ge=1), pageSize: int=Query(20,ge=1,le=100)):
    if request.app.state.content_database is None:
        return envelope(request,items=[],total=0,page=page,pageSize=pageSize)
    with database(request) as db:
        rows=[r for r in db.execute('''SELECT c.* FROM cases c JOIN landslides l ON l.id=c.landslide_id
            WHERE c.status='ready' AND (? IS NULL OR l.type=?) ORDER BY c.id''',(type,type))
            if r['landslide_id'] in visible_ids(request)]
        start=(page-1)*pageSize
        return envelope(request,items=[case_view(db,r,request) for r in rows[start:start+pageSize]],
                        total=len(rows),page=page,pageSize=pageSize)


@router.get('/cases/{case_id}')
def case_detail(case_id: str, request: Request):
    with database(request) as db:
        row=db.execute("SELECT * FROM cases WHERE id=? AND status='ready'",(case_id,)).fetchone()
        if row is None or row['landslide_id'] not in visible_ids(request):
            fail(404,'NOT_FOUND','Published case was not found.')
        return envelope(request,item=case_view(db,row,request))


@router.get('/documents')
def documents(request: Request):
    if request.app.state.content_database is None:return envelope(request,items=[],total=0)
    with database(request) as db:
        cited=set()
        for row in db.execute("SELECT id,landslide_id FROM cases WHERE status='ready'"):
            if row['landslide_id'] in visible_ids(request):
                cited.update(r['id'] for r in source_docs(db,row['id']))
        for row in db.execute("SELECT type FROM atlas_types WHERE status='ready'"):
            payload=content_payload(db,'type',row['type']) or {}
            cited.update(c['documentId'] for c in payload.get('citations',[]))
        rows=[dict(r) for r in db.execute('SELECT id,title,url,citation FROM documents ORDER BY id') if r['id'] in cited]
        return envelope(request,items=rows,total=len(rows))


@router.get('/documents/{document_id}')
def document(document_id: str, request: Request):
    result=documents(request)
    item=next((d for d in result['items'] if d['id']==document_id),None)
    if item is None:fail(404,'NOT_FOUND','Document was not found.')
    return envelope(request,item=item)


def find_asset(db, asset_id, request):
    row=db.execute('SELECT * FROM assets WHERE id=?',(asset_id,)).fetchone()
    if row is None or row['landslide_id'] not in visible_ids(request):
        fail(404,'NOT_FOUND','Resource was not found.')
    return row


def resource_path(row, request):
    root=request.app.state.asset_root
    if root is None:fail(503,'ASSET_ROOT_NOT_CONFIGURED','Resource root is not configured.')
    path=Path(row['path'])
    path=(path if path.is_absolute() else root/path).resolve()
    if not path.is_relative_to(root):fail(404,'RESOURCE_UNAVAILABLE','Resource is unavailable.')
    if not path.is_file():fail(404,'RESOURCE_UNAVAILABLE','Resource is unavailable.')
    return path


@router.get('/assets/{asset_id}')
def asset_metadata(asset_id: str, request: Request):
    with database(request) as db:
        row=find_asset(db,asset_id,request)
        return envelope(request,item=dict(id=row['id'],landslideId=row['landslide_id'],kind=row['kind'],
            width=row['width'],height=row['height'],
            contentUrl=f'/api/assets/{asset_id}/content' if row['kind']=='image' else None))


@router.get('/assets/{asset_id}/content')
def image_content(asset_id: str, request: Request):
    with database(request) as db:
        row=find_asset(db,asset_id,request)
        if row['kind']!='image':fail(415,'UNSUPPORTED_RESOURCE','Only preview images can be downloaded.')
        path=resource_path(row,request)
        if path.stat().st_size>16*1024*1024:fail(413,'RESOURCE_TOO_LARGE','Preview exceeds the size limit.')
        data=path.read_bytes()
        if row['sha256'] and hashlib.sha256(data).hexdigest()!=row['sha256']:
            fail(409,'RESOURCE_CHANGED','Resource does not match its registered hash.')
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format not in ('PNG','JPEG') or image.width*image.height>16777216:
                    fail(415,'UNSUPPORTED_RESOURCE','Preview must be a bounded PNG or JPEG.')
                mime='image/png' if image.format=='PNG' else 'image/jpeg'
                image.verify()
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
            fail(415,'UNSUPPORTED_RESOURCE','Preview could not be decoded.')
        return Response(data,media_type=mime,headers={'X-Content-Type-Options':'nosniff'})


@router.get('/landslides/{slide_id}/geometry')
def geometry(slide_id: str, request: Request, kind: Literal['point','polygon']='polygon'):
    if slide_id not in visible_ids(request):fail(404,'NOT_FOUND','Landslide was not found.')
    with database(request) as db:
        link=db.execute('SELECT * FROM geometry_links WHERE landslide_id=? AND kind=?',(slide_id,kind)).fetchone()
        if link is None:fail(404,'NOT_FOUND','Geometry was not found.')
        asset=find_asset(db,link['asset_id'],request)
        if asset['landslide_id']!=slide_id:
            fail(409,'GEOMETRY_MISMATCH','Geometry belongs to a different record.')
        path=resource_path(asset,request)
        if asset['kind']!=kind or path.suffix.lower()!='.shp':
            fail(415,'UNSUPPORTED_RESOURCE','Registered geometry is not a shapefile.')
        for extension in ('.shp','.shx','.dbf'):
            sidecar=path.with_suffix(extension).resolve()
            if not sidecar.is_relative_to(request.app.state.asset_root) or not sidecar.is_file():
                fail(404,'RESOURCE_UNAVAILABLE','Geometry is unavailable.')
        try:
            import shapefile
        except ImportError:
            fail(503,'GEOMETRY_READER_UNAVAILABLE','Geometry reader is not installed.')
        try:
            with shapefile.Reader(str(path),encoding='utf-8') as reader:
                index=link['record_index0']
                if index<0 or index>=len(reader):fail(409,'GEOMETRY_MISMATCH','Geometry index is invalid.')
                shape=reader.shape(index)
                if len(shape.points)>200000:fail(413,'RESOURCE_TOO_LARGE','Geometry exceeds the vertex limit.')
                record=reader.record(index).as_dict()
                slide=next(r for r in request.app.state.catalog.records if r.id==slide_id)
                sid=record.get('source_id',record.get('序号'))
                if sid is not None and slide.sourceId is not None and float(sid)!=slide.sourceId:
                    fail(409,'GEOMETRY_MISMATCH','Geometry belongs to a different record.')
                geom=shape.__geo_interface__
                expected=('Point','MultiPoint') if kind=='point' else ('Polygon','MultiPolygon')
                if geom['type'] not in expected:fail(409,'GEOMETRY_MISMATCH','Geometry type does not match its link.')
        except (OSError,ValueError,IndexError,shapefile.ShapefileException):
            fail(409,'GEOMETRY_INVALID','Registered geometry could not be read.')
        info=json.loads(asset['metadata_json'])
        # The original WGS84 PRJ is not evidence of a Mars latitude convention.
        convention=info.get('latitudeConvention','unverified')
        return envelope(request,coordinateSystem=dict(body='Mars',longitude='east-positive, -180..180',latitude=convention),
            canOverlay=convention=='planetocentric',
            feature=dict(type='Feature',id=slide_id,geometry=geom,
                         properties=dict(kind=kind,rawType=link['raw_type'])))
