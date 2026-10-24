from io import BytesIO
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import create_app
from api.sqlite_store import build_database,load_sqlite_catalog
from api.image_policy import decode_image,ImageInputError,MAX_UPLOAD_BYTES

DEMO=Path(__file__).resolve().parents[2]/'examples/catalog.demo.json'


def test_sqlite_matches_existing_catalog_and_query_statistics(tmp_path):
    payload=json.loads(DEMO.read_text(encoding='utf-8'));db=tmp_path/'catalog.sqlite3'
    build_database(db,payload)
    assert load_sqlite_catalog(db).model_dump()==payload
    with TestClient(create_app(database_path=db)) as client:
        assert client.get('/api/catalog').json()==payload
        query={'type':'SP','q':'demo'}
        assert client.get('/api/landslides',params=query).json()['total']==client.get('/api/stats',params=query).json()['total']
        assert client.get('/api/landslides/'+payload['records'][0]['id']).json()==payload['records'][0]
        assert client.get('/api/landslides/absent').status_code==404


def test_raw_inventory_stays_staged_and_original_types_survive(tmp_path):
    rows=[dict(id='ls_0059',source_id=59,type='SL',lon=-10.,lat=3.,area=4.,perimeter=2.)]
    db=tmp_path/'raw.sqlite3'
    build_database(db,rows,raw=True,source_index=[{'source_id':59,'raw_type':'RL'}])
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT raw_type,type FROM landslides').fetchone()==('RL','SL')
        conn.execute('PRAGMA foreign_keys=ON')
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO cases(id,landslide_id,title) VALUES('bad','missing','bad')")
    with pytest.raises(ValueError,match='staged'):load_sqlite_catalog(db)
    with pytest.raises(FileExistsError):build_database(db,rows,raw=True)


def test_duplicate_raw_ids_do_not_create_partial_database(tmp_path):
    row=dict(id='a',source_id=1,type='SL',lon=0.,lat=0.,area=None)
    db=tmp_path/'bad.sqlite3'
    with pytest.raises(ValueError):build_database(db,[row,row],raw=True)
    assert not db.exists()


def test_explicit_catalog_overrides_database_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('MARS_DB_PATH',str(tmp_path/'absent.sqlite3'))
    with TestClient(create_app(DEMO)) as client:assert client.get('/api/catalog').json()==json.loads(DEMO.read_text(encoding='utf-8'))
    with pytest.raises(ValueError):create_app(DEMO,database_path=tmp_path/'other.sqlite3')


def test_geometry_association_requires_existing_asset_and_target(tmp_path):
    payload=json.loads(DEMO.read_text(encoding='utf-8'));db=tmp_path/'geo.sqlite3';build_database(db,payload)
    with sqlite3.connect(db) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO geometry_links VALUES(?,'point','missing',0,'SL')",(payload['records'][0]['id'],))


def test_corrupt_source_reference_is_rejected_instead_of_serving_partial_catalog(tmp_path):
    payload=json.loads(DEMO.read_text(encoding='utf-8'));db=tmp_path/'corrupt.sqlite3';build_database(db,payload)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE landslides SET source_ref='missing' WHERE id=?",(payload['records'][0]['id'],))
    with pytest.raises(ValueError,match='foreign-key'):load_sqlite_catalog(db)


def image_bytes(size=(10,10),format='PNG'):
    out=BytesIO();Image.new('L',size,123).save(out,format=format);return out.getvalue()


def test_valid_grayscale_upload_and_limits():
    result=decode_image(image_bytes());assert result.mode=='RGB' and result.size==(10,10)
    for payload,code,status in [(b'','EMPTY_IMAGE',422),(b'not an image','INVALID_IMAGE',422),
        (b'x'*(MAX_UPLOAD_BYTES+1),'UPLOAD_TOO_LARGE',413),(image_bytes(format='GIF'),'UNSUPPORTED_IMAGE',415),
        (image_bytes((8193,1)),'IMAGE_DIMENSIONS_EXCEEDED',413),
        (image_bytes((4097,4096)),'IMAGE_DIMENSIONS_EXCEEDED',413)]:
        with pytest.raises(ImageInputError) as error:decode_image(payload)
        assert (error.value.code,error.value.status)==(code,status)
