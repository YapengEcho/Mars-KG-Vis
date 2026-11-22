"""Synthetic-only authentication, disclosure and request-isolation checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pickle
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.agent import llm
from api.agent.recognition import query_recognition_task
from api.classifier import classification_result, prepare_image
from api.main import create_app
from api.recognition_access import RecognitionAccessError, bind_task_context


ENDPOINTS = ['/api/agent/chat', '/api/agent/chat/stream']
SCORES = dict(SL=.1, SP=.2, FL=.3, CL=.4)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ('MARS_CLASSIFIER_MANIFEST', 'MARS_BINARY_SCREEN_MANIFEST', 'MARS_RECOGNITION_DB', 'MARS_DB_PATH',
                 'MARS_CATALOG_PATH', 'MARS_DOCS_PATH', 'DEEPSEEK_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    # Task lookup is a backend handshake, not a provider call or natural-language feature.
    monkeypatch.setattr(llm, 'is_configured', lambda: True)
    monkeypatch.setattr(llm, 'get_client', lambda: pytest.fail('Unexpected language model call'))


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(recognition_database=tmp_path/'synthetic.sqlite3')) as c:
        yield c


def device(client):
    return client.post('/api/recognition/sessions').json()['token']


def headers(token):
    return {'Authorization': 'Bearer ' + token}


def record(client, token, state='completed', scores=None, *, screening=None):
    store = client.app.state.recognition.store
    owner = store.authenticate(token)
    model = SimpleNamespace(version='synthetic-test-v1', asset_digest='0'*64)
    task_id = store.enqueue(owner, prepare_image(Image.new('RGB', (32, 24))), model.version)
    if state != 'queued':
        with store.connect() as db:
            db.execute("UPDATE recognition_jobs SET status='running' WHERE id=?", (task_id,))
    if state == 'completed':
        if screening is None:
            store.finish(task_id, result=classification_result(scores or SCORES, model))
        else:
            store.finish(task_id, screening=screening)
    elif state == 'failed':
        store.finish(task_id, error='INFERENCE_FAILED')
    return task_id


def detection_candidate(index=1, scores=None):
    return {
        'candidateId': str(index),
        'box': [4, 6, 20, 18],
        'detectionScore': .71,
        'classification': classification_result(
            scores or SCORES, SimpleNamespace(version='synthetic-test-v1', asset_digest='0' * 64)),
    }


def detection_record(client, token, *, candidates=None, error=None):
    """Create a synthetic detection task (kind=detect) owned by ``token``."""
    store = client.app.state.recognition.store
    owner = store.authenticate(token)
    task_id = store.enqueue_detection(owner, prepare_image(Image.new('RGB', (32, 24))),
                                      'synthetic-detector-v1')
    with store.connect() as db:
        db.execute("UPDATE recognition_jobs SET status='running' WHERE id=?", (task_id,))
    if error:
        store.finish(task_id, error=error)
    else:
        items = candidates or []
        store.finish(task_id, result={
            'outcome': 'detections' if items else 'no_detections',
            'coordinateSpace': 'image_pixels',
            'truncated': False,
            'candidates': items,
        })
    return task_id


def payload(task_id=None, **extra):
    data = {'messages': [{'role': 'user', 'content': '解释这条记录'}], **extra}
    if task_id is not None:
        data['recognitionTaskId'] = task_id
    return data


def final_response(response, endpoint):
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    if endpoint.endswith('/stream'):
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert events[-1]['type'] in ('done', 'error')
        assert not any(e['type'] == 'delta' for e in events)
        return events[-1]['response']
    return response.json()


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('state', ['queued', 'running', 'completed', 'failed'])
def test_authorized_lookup_state_and_allowlist(client, endpoint, state, caplog):
    token = device(client)
    task_id = record(client, token, state)
    result = final_response(client.post(endpoint, headers=headers(token), json=payload(task_id)), endpoint)
    assert result['status'] == 'ok'
    summary = result['recognitionTask']
    assert summary['taskId'] == task_id and summary['status'] == state
    assert set(summary) == {'taskId', 'kind', 'status', 'createdAt', 'finishedAt',
                            'modelVersion', 'preprocessingVersion', 'errorCode', 'result', 'screening'}
    assert summary['screening'] is None
    assert result['actions'] == [dict(type='open_task', taskId=task_id, slideId=None)]
    assert result['toolCalls'] == [dict(name='get_recognition_task', args={'taskId': task_id}, ok=True)]
    if state == 'completed':
        assert summary['result']['probabilities'] == SCORES
        source = client.get('/api/recognition/tasks/'+task_id, headers=headers(token)).json()
        assert summary['result'] == {k: v for k, v in source['result'].items() if k != 'modelDigest'}
        assert summary['result']['uncertain'] and not summary['result']['scoresCalibrated']
        assert '仅做四分类的历史记录' in result['answer']
    else:
        assert summary['result'] is None
    serialized = json.dumps(result) + repr(client.app.state.tracer.list()) + caplog.text
    for private in (token, hashlib.sha256(token.encode()).hexdigest(), 'owner_hash', 'modelDigest', 'imageUrl'):
        assert private not in serialized
    assert not client.get('/api/capabilities').json()['classifier']['available']  # No weights needed.


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_screened_out_recognition_can_be_read_without_inventing_a_type(client, endpoint):
    token = device(client)
    screening = dict(decision='no_landslide', score=-.25, threshold=.5, modelVersion='synthetic-screen-v1')
    task_id = record(client, token, screening=screening)
    response = client.post(endpoint, headers=headers(token), json=payload(task_id))
    data = final_response(response, endpoint)
    assert data['status'] == 'ok'
    assert data['recognitionTask']['status'] == 'completed'
    assert data['recognitionTask']['screening'] == screening
    assert data['recognitionTask']['result'] is None
    assert '四分类' in data['answer'] and '候选类型' not in data['answer']
    foreign = client.post(endpoint, headers=headers(device(client)), json=payload(task_id))
    assert foreign.status_code == 404


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('authorization', [None, 'Basic abc', 'Bearer short', 'Bearer '+'a'*43])
def test_missing_malformed_or_unknown_token_fails_before_agent(client, endpoint, authorization):
    token = device(client)
    task_id = record(client, token)
    response = client.post(endpoint, json=payload(task_id, sessionId=token),
                           headers={} if authorization is None else {'Authorization': authorization})
    assert response.status_code == 401
    assert response.json()['detail']['code'] == 'AUTH_REQUIRED'
    assert response.headers['www-authenticate'] == 'Bearer'
    assert response.headers['cache-control'] == 'no-store'
    assert 'text/event-stream' not in response.headers['content-type']
    assert client.app.state.tracer.list() == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_expired_token_is_not_restored_by_session_id(client, endpoint):
    token = device(client)
    task_id = record(client, token)
    with client.app.state.recognition.store.connect() as db:
        db.execute("UPDATE recognition_sessions SET expires_at='2000-01-01'")
    response = client.post(endpoint, json=payload(task_id, sessionId=task_id), headers=headers(token))
    assert response.status_code == 401
    assert client.app.state.tracer.list() == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_foreign_missing_and_deleted_are_indistinguishable(client, endpoint):
    a, b = device(client), device(client)
    task_id = record(client, a)
    foreign = client.post(endpoint, json=payload(task_id, sessionId=a), headers=headers(b))
    missing = client.post(endpoint, json=payload('f'*32), headers=headers(b))
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    assert client.delete('/api/recognition/tasks/'+task_id, headers=headers(a)).status_code == 204
    deleted = client.post(endpoint, json=payload(task_id), headers=headers(a))
    assert deleted.status_code == 404 and deleted.json() == missing.json()
    assert client.app.state.tracer.list() == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('bad_id', ['short', 'A'*32, 'f'*32+'\n', '../tasks/a', 123, {'taskId': 'f'*32}])
def test_bad_task_id_never_enters_agent(client, endpoint, bad_id):
    response = client.post(endpoint, json=payload(bad_id), headers=headers(device(client)))
    assert response.status_code == 422
    assert response.headers['cache-control'] == 'no-store'
    assert client.app.state.tracer.list() == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_body_cannot_supply_owner_token_or_result(client, endpoint):
    token = device(client)
    task_id = record(client, token)
    response = client.post(endpoint, headers=headers(token), json=payload(task_id,
        owner='attacker', token='body-token', recognitionContext={}, result={'prediction': 'SL'}))
    assert response.status_code == 422
    assert client.app.state.tracer.list() == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_store_unavailable_is_sanitized(client, endpoint, monkeypatch, caplog):
    token = device(client)
    task_id = record(client, token)
    def broken(*args):
        raise sqlite3.OperationalError('private/path ' + token)
    monkeypatch.setattr(client.app.state.recognition.store, 'get', broken)
    response = client.post(endpoint, json=payload(task_id), headers=headers(token))
    assert response.status_code == 503 and response.json()['detail']['code'] == 'RECOGNITION_UNAVAILABLE'
    assert token not in response.text + caplog.text
    assert client.app.state.tracer.list() == []


def test_authentication_storage_failure_and_missing_credentials(client, monkeypatch):
    token = device(client)
    def broken(*args):
        raise sqlite3.OperationalError(token)
    monkeypatch.setattr(client.app.state.recognition.store, 'authenticate', broken)
    for endpoint in ENDPOINTS + ['/api/recognition/tasks/'+'f'*32]:
        call = (lambda h: client.get(endpoint, headers=h)) if '/recognition/tasks/' in endpoint else (
            lambda h: client.post(endpoint, json=payload('f'*32), headers=h))
        assert call({}).status_code == 401
        response = call(headers(token))
        assert response.status_code == 503 and token not in response.text


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_public_chat_works_without_recognition_store(monkeypatch, endpoint):
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='普通问答', tool_calls=None))])])
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    with TestClient(create_app()) as c:
        response = c.post(endpoint, json=payload())
        # Ordinary chat remains streamed text; no task summaries or authorization needed.
        assert response.status_code == 200
        if endpoint.endswith('/stream'):
            data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')][-1]['response']
        else:
            data = response.json()
        assert data['answer'] == '普通问答' and data['recognitionTask'] is None
        assert len(captured) == 1
        assert 'get_recognition_task' not in json.dumps(captured)
        assert c.post(endpoint, json=payload('f'*32)).status_code == 401
        absent = c.post(endpoint, json=payload('f'*32), headers=headers('a'*43))
        assert absent.status_code == 503 and absent.json()['detail']['code'] == 'RECOGNITION_NOT_CONFIGURED'
        assert len(captured) == 1


def test_context_is_single_task_unserializable_and_queries_fresh(client):
    token = device(client)
    a, b = record(client, token), record(client, token)
    store = client.app.state.recognition.store
    context = bind_task_context(store, token, a)
    assert context.read(a).taskId == a
    assert context.task_id == a
    assert token not in repr(context) and store.authenticate(token) not in repr(context)
    for serialize in (json.dumps, pickle.dumps):
        with pytest.raises(TypeError):
            serialize(context)
    with pytest.raises(AttributeError):
        context.task_id = b
    assert query_recognition_task(context, {'taskId': b}).payload['error']['code'] == 'TASK_NOT_FOUND'
    assert query_recognition_task(None, {'taskId': a}).payload['error']['code'] == 'AUTH_REQUIRED'
    forged = query_recognition_task(context, {'taskId': a, 'owner': token, 'url': 'https://invalid.example'})
    assert not forged.ok and forged.args == {} and token not in repr(forged)
    assert client.delete('/api/recognition/tasks/'+a, headers=headers(token)).status_code == 204
    with pytest.raises(RecognitionAccessError) as err:
        context.read(a)
    assert err.value.code == 'TASK_NOT_FOUND'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_deletion_between_preflight_and_lookup_does_not_serve_cached_result(client, endpoint, monkeypatch):
    import api.main as main
    token = device(client)
    task_id = record(client, token)
    function_name = 'iter_agent_events' if endpoint.endswith('/stream') else 'run_agent'
    original = getattr(main, function_name)
    def delete_then_run(*args, **kwargs):
        store = client.app.state.recognition.store
        store.delete(store.authenticate(token), task_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(main, function_name, delete_then_run)
    result = final_response(client.post(endpoint, json=payload(task_id), headers=headers(token)), endpoint)
    assert result['status'] == 'error' and result['recognitionTask'] is None
    assert result['actions'] == [] and not result['toolCalls'][0]['ok']


def test_concurrent_devices_do_not_share_context(client):
    a, b = device(client), device(client)
    a_id, b_id = record(client, a), record(client, b, scores=dict(SL=.7, SP=.1, FL=.1, CL=.1))
    cases = [(a, a_id, 'CL'), (b, b_id, 'SL')] * 6
    def request(case):
        token, task_id, expected = case
        response = client.post('/api/agent/chat', json=payload(task_id, sessionId='shared'), headers=headers(token))
        assert response.status_code == 200
        data = response.json()['recognitionTask']
        assert data['taskId'] == task_id and data['result']['prediction'] == expected
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(request, cases))


def test_summary_drops_extra_fields_and_rejects_corrupt_result(client, monkeypatch):
    token = device(client)
    task_id = record(client, token)
    store = client.app.state.recognition.store
    original = store.get
    def extra(*args):
        task = original(*args)
        task['owner_hash'] = 'private-owner'
        task['result']['privateField'] = 'private-param'
        return task
    monkeypatch.setattr(store, 'get', extra)
    context = bind_task_context(store, token, task_id)
    assert 'private' not in context.read(task_id).model_dump_json()
    def corrupt(*args):
        task = original(*args)
        task['result']['probabilities']['SP'] = float('nan')
        return task
    monkeypatch.setattr(store, 'get', corrupt)
    response = client.post('/api/agent/chat', json=payload(task_id), headers=headers(token))
    assert response.status_code == 503 and response.json()['detail']['code'] == 'RECOGNITION_UNAVAILABLE'


def test_upstream_exception_trace_is_redacted(client, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError('sensitive-upstream-key-and-path')
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=broken))))
    response = client.post('/api/agent/chat', json=payload())
    assert response.json()['status'] == 'error'
    traces = repr(client.app.state.tracer.list())
    assert 'sensitive-upstream' not in traces + response.text and 'AGENT_UPSTREAM_ERROR' in traces


def test_openapi_explains_conditional_auth_and_safe_summary():
    schema = create_app().openapi()
    for endpoint in ENDPOINTS:
        security = schema['paths'][endpoint]['post']['security']
        assert {} in security and {'HTTPBearer': []} in security
        assert {'401', '404', '503'} <= set(schema['paths'][endpoint]['post']['responses'])
    assert set(schema['paths'][ENDPOINTS[1]]['post']['responses']['200']['content']) == {'text/event-stream'}
    request = schema['components']['schemas']['ChatRequest']
    assert 'recognitionTaskId' in request['properties'] and request['additionalProperties'] is False
    assert 'recognitionTask' in schema['components']['schemas']['ChatResponse']['properties']
    assert 'modelDigest' not in schema['components']['schemas']['ResultSummary']['properties']


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_detection_task_lists_candidates_and_keeps_review_required(client, endpoint):
    """检测任务：列出候选框，且不把候选说成已确认的类型。"""
    token = device(client)
    task_id = detection_record(client, token, candidates=[detection_candidate(1), detection_candidate(2)])
    data = final_response(client.post(endpoint, headers=headers(token), json=payload(task_id)), endpoint)

    assert data['status'] == 'ok'
    summary = data['recognitionTask']
    assert summary['kind'] == 'detect' and summary['status'] == 'completed'
    # 白名单：不含图片、输入几何、筛选项与任何摘要
    assert set(summary) == {'taskId', 'kind', 'status', 'createdAt', 'finishedAt',
                            'modelVersion', 'preprocessingVersion', 'errorCode', 'result'}
    assert summary['result']['outcome'] == 'detections'
    assert len(summary['result']['candidates']) == 2
    assert summary['modelVersion'] == 'synthetic-detector-v1'

    answer = data['answer']
    assert '2 个候选框' in answer
    assert '待复核' in answer and '检测分' in answer
    assert '1' in answer and '2' in answer
    for claim in ('已确认', '确定是', '一定是'):
        assert claim not in answer
    assert data['actions'] == [dict(type='open_task', taskId=task_id, slideId=None)]

    serialized = json.dumps(data) + repr(client.app.state.tracer.list())
    for private in (token, 'modelDigest', 'candidateBox', 'imageUrl'):
        assert private not in serialized


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_detection_without_candidates_reports_not_detected(client, endpoint):
    """未检出：明确说没找到候选，不补写类型。"""
    token = device(client)
    task_id = detection_record(client, token, candidates=[])
    data = final_response(client.post(endpoint, headers=headers(token), json=payload(task_id)), endpoint)

    assert data['status'] == 'ok'
    assert data['recognitionTask']['result']['outcome'] == 'no_detections'
    assert data['recognitionTask']['result']['candidates'] == []
    answer = data['answer']
    assert '未检出' in answer
    assert '分类为' not in answer
    assert data['actions'] == [dict(type='open_task', taskId=task_id, slideId=None)]


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_detection_failure_reports_error_code_without_candidates(client, endpoint):
    """失败：给错误码，不给候选，也不编造结果。"""
    token = device(client)
    task_id = detection_record(client, token, error='INFERENCE_FAILED')
    data = final_response(client.post(endpoint, headers=headers(token), json=payload(task_id)), endpoint)

    summary = data['recognitionTask']
    assert summary['status'] == 'failed'
    assert summary['result'] is None
    assert summary['errorCode'] == 'INFERENCE_FAILED'
    assert '未完成' in data['answer'] and 'INFERENCE_FAILED' in data['answer']


def test_detection_task_of_another_device_is_not_found(client):
    owner = device(client)
    task_id = detection_record(client, owner, candidates=[detection_candidate()])
    other = device(client)
    response = client.post('/api/agent/chat', headers=headers(other), json=payload(task_id))
    assert response.status_code == 404 and response.json()['detail']['code'] == 'TASK_NOT_FOUND'


def test_detection_result_rejects_inconsistent_outcome():
    """有候选却标未检出（或反之）属于坏数据，不能当成正常结果返回。"""
    from pydantic import ValidationError

    from api.recognition_access import DetectionResult

    with pytest.raises(ValidationError):
        DetectionResult.model_validate({
            'outcome': 'no_detections',
            'coordinateSpace': 'image_pixels',
            'truncated': False,
            'candidates': [detection_candidate()],
        })
    with pytest.raises(ValidationError):
        DetectionResult.model_validate({
            'outcome': 'detections',
            'coordinateSpace': 'image_pixels',
            'truncated': False,
            'candidates': [],
        })
