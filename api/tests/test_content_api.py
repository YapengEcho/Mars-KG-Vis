"""Content publication, idempotent imports and resource-boundary regression tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import shapefile

from api.content_store import import_atlas
from api.main import create_app
from api.sqlite_store import build_database


@pytest.fixture
def content_fixture(tmp_path):
    source=json.loads((Path(__file__).parents[2]/'examples/catalog.demo.json').read_text(encoding='utf-8'))
    source['records']=source['records'][:4]
    for i,(row,kind) in enumerate(zip(source['records'],('SL','SP','FL','CL')),1):
        row.update(id=f'demo-{i}',sourceId=i,type=kind,longitude=10.,latitude=20.)
    db=tmp_path/'catalog.sqlite3';build_database(db,source)
    bundle=dict(version='synthetic-atlas-v1',sourceDocument=dict(id='test-doc',title='Synthetic source',url='https://example.com/paper'),
                types=[],cases=[],comparisons=[],limits=['Synthetic fixtures, not geological guidance.'])
    for i,kind in enumerate(('SL','SP','FL','CL'),1):
        cid=f'case-demo-{i}'
        bundle['types'].append(dict(type=kind,name=f'Demo {kind}',quote='Synthetic definition',caseIds=[cid],
            citations=[dict(documentId='test-doc',locator='Synthetic page 1')]))
        bundle['cases'].append(dict(id=cid,landslideId=f'demo-{i}',type=kind,title='Synthetic case',
            observations=['A synthetic observation'],citationLocator='Synthetic page 1'))
    bundle['comparisons']=[dict(id='sl-sp',left='SL',right='SP',title='Synthetic comparison',rows=[],citations=[])]
    content=tmp_path/'atlas.json';content.write_text(json.dumps(bundle),encoding='utf-8')
    return db,content,bundle


def test_default_json_mode_has_explicit_empty_content():
    with TestClient(create_app()) as client:
        assert client.get('/api/atlas').json()['types']==[]
        assert client.get('/api/cases').json()['total']==0
        assert client.get('/api/documents').json()['total']==0
        assert client.get('/api/cases/missing').status_code==404


def test_drafts_are_hidden_then_publish_enables_linked_content(content_fixture):
    db,content,bundle=content_fixture
    import_atlas(db,content)
    with TestClient(create_app(database_path=db)) as client:
        assert client.get('/api/atlas').json()['types']==[]
        assert client.get('/api/cases').json()['total']==0
        assert client.get('/api/cases/case-demo-1').status_code==404
        assert client.get('/api/documents').json()['total']==0
    import_atlas(db,content,publish=True)
    with TestClient(create_app(database_path=db)) as client:
        overview=client.get('/api/atlas').json()
        assert len(overview['types'])==4 and len(overview['comparisons'])==1
        assert overview['contentVersion']==bundle['version']
        assert client.get('/api/atlas/SL').json()['item']['content']['caseIds']==['case-demo-1']
        assert client.get('/api/atlas/unknown').status_code==422
        all_rows=client.get('/api/cases',params={'pageSize':2}).json()
        assert all_rows['total']==4 and len(all_rows['items'])==2
        filtered=client.get('/api/cases',params={'type':'SP'}).json()
        assert filtered['total']==1 and filtered['items'][0]['type']=='SP'
        item=client.get('/api/cases/case-demo-1').json()['item']
        assert item['landslide']['id']=='demo-1' and item['content']['observations']==['A synthetic observation']
        assert item['documents'][0]['id']=='test-doc' and item['documents'][0]['locator']=='Synthetic page 1'
        assert client.get('/api/documents/test-doc').json()['item']['url']=='https://example.com/paper'
        assert client.get('/api/documents/missing').status_code==404


def test_repeat_import_preserves_asset_links_publication_and_citations(content_fixture,tmp_path):
    db,content,_=content_fixture;import_atlas(db,content,publish=True)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO assets(id,landslide_id,kind,path) VALUES('img','demo-1','image','preview.png')")
        conn.execute("UPDATE cases SET image_asset_id='img' WHERE id='case-demo-1'")
    import_atlas(db,content)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT image_asset_id,status FROM cases WHERE id='case-demo-1'").fetchone()==('img','ready')
        assert conn.execute('SELECT count(*) FROM case_documents').fetchone()[0]==4
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()


@pytest.mark.parametrize('mutation',['missing-slide','wrong-type','bad-citation','duplicate-case'])
def test_invalid_import_rolls_back_existing_content(content_fixture,mutation):
    db,content,bundle=content_fixture;import_atlas(db,content,publish=True)
    with sqlite3.connect(db) as conn:
        before=conn.execute('SELECT * FROM cases ORDER BY id').fetchall()
        version=conn.execute("SELECT value FROM metadata WHERE key='atlasContentVersion'").fetchone()
    bad=deepcopy(bundle);bad['version']='must-not-appear'
    if mutation=='missing-slide':bad['cases'][0]['landslideId']='missing'
    if mutation=='wrong-type':bad['cases'][0]['type']='SP'
    if mutation=='bad-citation':bad['types'][0]['citations'][0]['documentId']='unknown'
    if mutation=='duplicate-case':bad['cases'].append(bad['cases'][0])
    content.write_text(json.dumps(bad),encoding='utf-8')
    with pytest.raises(ValueError):import_atlas(db,content,publish=True)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT * FROM cases ORDER BY id').fetchall()==before
        assert conn.execute("SELECT value FROM metadata WHERE key='atlasContentVersion'").fetchone()==version


def register_resources(db,root):
    image=root/'preview.png';Image.new('L',(20,20),120).save(image)
    with shapefile.Writer(str(root/'boundary'),shapeType=shapefile.POLYGON) as writer:
        writer.field('source_id','N');writer.poly([[[9,19],[9,21],[11,21],[11,19],[9,19]]]);writer.record(1)
    with sqlite3.connect(db) as conn:
        conn.execute('INSERT INTO assets(id,landslide_id,kind,path,sha256) VALUES(?,?,?,?,?)',
                     ('img','demo-1','image',str(image),hashlib.sha256(image.read_bytes()).hexdigest()))
        conn.execute('INSERT INTO assets(id,landslide_id,kind,path,metadata_json) VALUES(?,?,?,?,?)',
                     ('poly','demo-1','polygon',str(root/'boundary.shp'),'{}'))
        conn.execute("INSERT INTO geometry_links VALUES('demo-1','polygon','poly',0,'SL')")


def test_preview_geometry_and_no_filesystem_path_disclosure(content_fixture,tmp_path):
    db,content,_=content_fixture;import_atlas(db,content,publish=True)
    root=tmp_path/'resources';root.mkdir();register_resources(db,root)
    with TestClient(create_app(database_path=db,asset_root=root)) as client:
        info=client.get('/api/assets/img').json()
        assert 'path' not in info['item'] and str(root) not in json.dumps(info)
        image=client.get('/api/assets/img/content')
        assert image.status_code==200 and image.headers['content-type']=='image/png'
        geo=client.get('/api/landslides/demo-1/geometry').json()
        assert geo['feature']['geometry']['type']=='Polygon'
        assert geo['coordinateSystem']['body']=='Mars' and geo['canOverlay'] is False
        assert geo['coordinateSystem']['latitude']=='unverified'
        assert str(root) not in json.dumps(geo)
        assert client.get('/api/assets/poly/content').status_code==415
        assert client.get('/api/landslides/demo-1/geometry?kind=point').status_code==404
        assert client.get('/api/assets/absent/content').status_code==404
        (root/'preview.png').write_bytes(b'changed')
        assert client.get('/api/assets/img/content').status_code==409


def test_resource_root_and_geometry_identity_enforced(content_fixture,tmp_path):
    db,_,_=content_fixture;root=tmp_path/'resources';root.mkdir();register_resources(db,root)
    outside=tmp_path/'outside.png';Image.new('RGB',(2,2)).save(outside)
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE assets SET path=? WHERE id=?',(str(outside),'img'))
    with TestClient(create_app(database_path=db,asset_root=root)) as client:
        assert client.get('/api/assets/img/content').status_code==404
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE geometry_links SET record_index0=99")
        assert client.get('/api/landslides/demo-1/geometry').status_code==409
    with TestClient(create_app(database_path=db)) as client:
        assert client.get('/api/assets/img/content').status_code==503


def test_legacy_database_without_optional_content_table(content_fixture):
    db,_,_=content_fixture
    with sqlite3.connect(db) as conn:conn.execute('DROP TABLE atlas_content')
    with TestClient(create_app(database_path=db)) as client:
        assert client.get('/api/atlas').json()['types']==[]
        assert client.get('/api/landslides').json()['total']==4


def test_case_rejects_image_linked_to_a_different_slide(content_fixture,tmp_path):
    db,content,_=content_fixture;import_atlas(db,content,publish=True)
    root=tmp_path/'resources';root.mkdir();register_resources(db,root)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE assets SET landslide_id='demo-2' WHERE id='img'")
        conn.execute("UPDATE cases SET image_asset_id='img' WHERE id='case-demo-1'")
    with TestClient(create_app(database_path=db,asset_root=root)) as client:
        assert client.get('/api/cases/case-demo-1').status_code==409
