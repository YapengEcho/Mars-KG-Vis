import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api.agent import llm
from api.agent.records import TraceRecorder
from api.agent.schemas import ChatRequest
from api.agent.service import run_agent
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


def test_catalog_tool_keeps_coordinate_metadata_out_of_model_context():
    catalog=load_public_catalog()
    retriever = KeywordRetriever([])
    for name, args in [
        ('get_slide', {'id': 'ls_0493'}),
        ('search_catalog', {'q': '493'}),
        ('locate_slide', {'id': 'ls_0493'}),
        ('get_stats', {}),
    ]:
        result = execute_tool(name, args, catalog, retriever)
        assert result.ok and result.payload['dataMode'] == 'reference'
        if name != 'get_stats':
            assert result.payload['coordinateSystem']['latitude'] == 'unverified, degrees'
        assert result.model_payload.keys().isdisjoint({'dataMode', 'coordinateSystem', 'version'})

    detail = execute_tool('get_slide', {'id': 'ls_0493'}, catalog, retriever)
    assert detail.model_payload['latitude'] == catalog.records[0].latitude
    locate = execute_tool('locate_slide', {'id': 'ls_0493'}, catalog, retriever)
    assert locate.model_payload == {'located': 'ls_0493'}
    assert locate.citations == detail.citations
    stats = execute_tool('get_stats', {}, catalog, retriever)
    assert stats.model_payload['total'] == 8


def test_language_model_round_trip_uses_public_catalog_facts(monkeypatch):
    received_messages = []
    tool_call = SimpleNamespace(index=0, id='get-493', function=SimpleNamespace(
        name='get_slide', arguments='{"id":"493"}',
    ))
    deltas = iter([
        SimpleNamespace(content=None, tool_calls=[tool_call]),
        SimpleNamespace(content='这处滑坡属于滑移型。', tool_calls=None),
    ])

    def complete(*, messages, **kwargs):
        received_messages.append(tuple(messages))
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=next(deltas))])])

    monkeypatch.setattr(llm, 'is_configured', lambda: True)
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=complete),
    )))
    request = ChatRequest(messages=[{'role': 'user', 'content': '请分析滑坡 493 的地貌环境。'}])
    response = run_agent(request, load_public_catalog(), KeywordRetriever([]), TraceRecorder())
    assert response.status == 'ok'
    tool_message = received_messages[1][-1]
    assert tool_message['role'] == 'tool'
    facts = json.loads(tool_message['content'])
    assert facts['type'] == 'SL' and facts['longitude'] is not None
    assert facts.keys().isdisjoint({'dataMode', 'coordinateSystem', 'version'})
    assert response.citations[0].url is not None
