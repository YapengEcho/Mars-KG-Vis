import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api.main import create_app
from api.public_catalog import load_public_catalog
from api.schemas import Catalog
from api.agent.tools import execute_tool
from api.agent.retrieval import KeywordRetriever


def test_reference_preserves_coordinates_and_cannot_be_promoted():
    catalog = load_public_catalog()
    source = json.loads((Path(__file__).parents[2]/'examples/catalog.real-subset.json').read_text(encoding='utf-8'))
    assert catalog.mode == 'reference' and len(catalog.records) == 8
    assert [(r.sourceId,r.longitude,r.latitude) for r in catalog.records] == [(r['sourceId'],r['longitude'],r['latitude']) for r in source['records']]
    assert all(not r.isDemo for r in catalog.records)
    value=catalog.model_dump();value['mode']='catalog'
    with pytest.raises(ValidationError):Catalog.model_validate(value)


def test_default_app_and_image_endpoint_are_real_public_subset(monkeypatch):
    monkeypatch.delenv('MARS_CATALOG_PATH',raising=False)
    monkeypatch.delenv('MARS_DB_PATH',raising=False)
    with TestClient(create_app()) as client:
        catalog=client.get('/api/catalog').json()
        assert catalog['mode']=='reference'
        assert len(catalog['records'])==8
        assert client.get('/api/landslides',params={'q':'493'}).json()['items'][0]['id']=='ls_0493'
        image=client.get(catalog['records'][0]['imageUrl'])
        assert image.status_code==200 and image.headers['content-type']=='image/png'
        assert client.get('/samples/classifier-manifest.json').status_code==404


def test_agent_receives_reference_coordinate_status():
    catalog=load_public_catalog()
    for name,args in [('get_slide',{'id':'ls_0493'}),('search_catalog',{'q':'493'}),('locate_slide',{'id':'ls_0493'})]:
        result=execute_tool(name,args,catalog,KeywordRetriever([]))
        assert result.ok and result.payload['dataMode']=='reference'
        assert result.payload['coordinateSystem']['latitude']=='unverified, degrees'
