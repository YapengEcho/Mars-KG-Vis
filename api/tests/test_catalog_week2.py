"""Spatial HTTP/Agent parity and import lineage regression checks."""
import hashlib
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

from api.agent.retrieval import KeywordRetriever
from api.agent.tools import execute_tool
from api.catalog_query import parse_bbox
from api.main import create_app
from api.schemas import Catalog
from api.sqlite_store import build_database, load_sqlite_catalog


def fixture_catalog():
    source = json.loads((Path(__file__).parents[2] / 'examples/catalog.demo.json').read_text(encoding='utf-8'))
    template = source['records'][0]
    # Synthetic positions include both spellings of the seam and a boundary.
    source['records'] = [dict(template, id=f'demo-{i}', sourceId=i, name='same query',
        type=t, longitude=float(lon), latitude=float(lat))
        for i, (lon,lat,t) in enumerate([(175,0,'SL'),(-175,0,'SP'),(0,0,'SL'),
            (180,5,'SL'),(-180,5,'SP'),(170,10,'FL'),(175,11,'CL')], 1)]
    return source


@pytest.mark.parametrize('bounds,expected', [
    ('170,-10,-170,10', [1,2,4,5,6]),
    ('-10,-10,10,10', [3]),
    ('-180,-90,180,90', [1,2,3,4,5,6,7]),
    ('180,5,180,5', [4,5]),
    ('-180,5,-180,5', [4,5]),
])
def test_sqlite_http_and_agent_same_spatial_query(tmp_path,bounds,expected):
    payload=fixture_catalog(); path=tmp_path/'catalog.sqlite3';build_database(path,payload)
    catalog=load_sqlite_catalog(path)
    with TestClient(create_app(database_path=path)) as client:
        query={'bbox':bounds,'q':'same','pageSize':2}
        listed=client.get('/api/landslides',params=query).json()
        stats=client.get('/api/stats',params={'bbox':bounds,'q':'same'}).json()
        rows=[]
        for page in range(1, (listed['total']+1)//2+1):
            rows.extend(client.get('/api/landslides',params=dict(query,page=page)).json()['items'])
        assert [r['sourceId'] for r in rows] == expected
        tool=execute_tool('search_catalog',{'q':'same','bbox':bounds,'limit':100},catalog,KeywordRetriever([]))
        counts=execute_tool('get_stats',{'q':'same','bbox':bounds},catalog,KeywordRetriever([]))
        assert tool.ok and counts.ok
        assert [r.sourceId for r in tool.records] == expected
        assert listed['total'] == stats['total'] == counts.payload['total'] == len(expected)
        assert stats['byType'] == counts.payload['byType']
        assert stats['version'] == tool.payload['version'] == counts.payload['version']
        typed=client.get('/api/landslides',params={'bbox':bounds,'type':'SL','pageSize':100}).json()
        assert all(r['type']=='SL' and r['sourceId'] in expected for r in typed['items'])


@pytest.mark.parametrize('value',['','1,2,3','1,2,3,4,5','nan,0,1,2','0,0,inf,2',
    '-181,0,0,1','0,-91,0,1','0,2,0,1','x,0,1,2'])
def test_bad_bounds_fail_consistently(tmp_path,value):
    payload=fixture_catalog(); path=tmp_path/'catalog.json';path.write_text(json.dumps(payload),encoding='utf-8')
    with TestClient(create_app(path)) as client:
        for endpoint in ['/api/landslides','/api/stats']:
            assert client.get(endpoint,params={'bbox':value}).status_code == 422
    assert not execute_tool('search_catalog',{'bbox':value},Catalog.model_validate(payload),KeywordRetriever([])).ok


def test_agent_pagination_has_same_order_and_rejects_bad_arguments():
    catalog=Catalog.model_validate(fixture_catalog());retriever=KeywordRetriever([])
    result=execute_tool('search_catalog',{'offset':2,'limit':2},catalog,retriever)
    assert [r.sourceId for r in result.records] == [3,4]
    assert result.payload['total'] == 7
    for args in [{'offset':-1},{'limit':0},{'limit':True},{'limit':'5'},{'type':'bad'},{'q':3},{'bbox':[]}]:
        assert not execute_tool('search_catalog',args,catalog,retriever).ok


def test_import_provenance_and_original_aliases_survive(tmp_path):
    payload=fixture_catalog(); path=tmp_path/'catalog.sqlite3'
    build_database(path,payload,source_index=[{'source_id':1,'raw_type':'RL'}],
        version_override='week2-test',provenance={'input_file_sha256':'test-only'})
    with sqlite3.connect(path) as db:
        metadata={k:json.loads(v) for k,v in db.execute('SELECT * FROM metadata')}
        assert metadata['version']=='week2-test'
        assert metadata['importProvenance']['input_file_sha256']=='test-only'
        expected=hashlib.sha256(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        assert metadata['importProvenance']['payload_sha256']==expected
        assert db.execute('SELECT raw_type,type FROM landslides WHERE source_id=1').fetchone()==('RL','SL')
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
    assert load_sqlite_catalog(path).version == 'week2-test'


def test_staging_keeps_original_latitude_and_rejects_serving(tmp_path):
    payload=[dict(id='ls_0001',source_id=1,type='SL',lon=10.,lat=45.,area=None)]
    path=tmp_path/'staging.sqlite3';build_database(path,payload,raw=True,version_override='unverified-v2')
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT latitude FROM landslides').fetchone()[0]==45.
    with pytest.raises(ValueError,match='staged'):load_sqlite_catalog(path)
