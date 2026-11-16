from io import BytesIO
import json
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.classifier import CLASSES, PREPROCESSING_VERSION, classification_result, load_manifest, prepare_image, sha256
from api.main import create_app
from api.recognition_store import QueueFull, TaskStore
from api.recognition_worker import WorkerLock
from api.recognition_routes import RecognitionBoundary


class FakeClassifier:
    version = 'synthetic-test-v1'
    asset_digest = '0'*64

    def predict(self, prepared):
        assert len(prepared.pixels) == 224*224
        assert sum(prepared.weights) == pytest.approx(1)
        return dict(SL=.1, SP=.2, FL=.3, CL=.4)


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    for name in ('MARS_CLASSIFIER_MANIFEST', 'MARS_RECOGNITION_DB', 'MARS_DB_PATH'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(classifier=FakeClassifier(), recognition_database=tmp_path/'tasks.sqlite3')) as c:
        yield c


def credential(client):
    response = client.post('/api/recognition/sessions')
    assert response.status_code == 201
    assert response.headers['cache-control'] == 'no-store'
    return {'Authorization': 'Bearer '+response.json()['token']}


def png(size=(280,140)):
    stream = BytesIO()
    Image.new('RGB', size, (20,70,100)).save(stream, format='PNG')
    return stream.getvalue()


def submit(client, headers, data=None, payload=None):
    return client.post('/api/landslide/classify', headers=headers, data=data,
                       files={'file': ('sample.png', payload if payload is not None else png(), 'image/png')})


def wait_result(client, headers, task_id):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        task = client.get('/api/recognition/tasks/'+task_id, headers=headers).json()
        if task['status'] in ('completed','failed'):
            return task
        time.sleep(.02)
    raise AssertionError('Synthetic inference did not complete')


def test_task_lifecycle_isolation_image_delete_and_version(client):
    a, b = credential(client), credential(client)
    capability = client.get('/api/capabilities').json()['classifier']
    assert capability['available'] and capability['historyAvailable']
    assert capability['preprocessingVersion'] == PREPROCESSING_VERSION
    response = submit(client,a,{'roi':'[20,20,200,120]'})
    assert response.status_code == 202
    task_id = response.json()['taskId']
    url = '/api/recognition/tasks/'+task_id
    assert response.headers['location'] == url
    assert response.headers['cache-control'] == 'no-store'
    task = wait_result(client,a,task_id)
    assert task['status'] == 'completed'
    assert task['input']['roi'] == [20,20,200,120]
    assert task['input']['roiSource'] == 'provided_box'
    assert task['result']['prediction'] == 'CL'
    assert task['result']['uncertain'] is True and task['result']['scoresCalibrated'] is False
    assert task['result']['modelVersion'] == task['modelVersion'] == FakeClassifier.version
    assert set(task['result']['probabilities']) == set(CLASSES)
    assert 'owner_hash' not in json.dumps(task)
    assert client.get('/api/recognition/tasks',headers=a).json()['total'] == 1
    assert client.get('/api/recognition/tasks',headers=b).json()['items'] == []
    for suffix in ('','/image'):
        missing = client.get('/api/recognition/tasks/absent'+suffix,headers=b)
        foreign = client.get(url+suffix,headers=b)
        assert missing.status_code == foreign.status_code == 404
        assert missing.json() == foreign.json()
        assert client.get(url+suffix).status_code == 401
    assert client.delete(url,headers=b).status_code == 404
    image = client.get(url+'/image',headers=a)
    assert image.status_code == 200 and image.headers['content-type'] == 'image/jpeg'
    assert image.headers['cache-control'] == 'no-store'
    assert Image.open(BytesIO(image.content)).size == (280,140)
    assert client.delete(url,headers=a).status_code == 204
    assert client.get(url,headers=a).status_code == 404
    assert client.get(url+'/image',headers=a).status_code == 404


def test_persistence_and_history_available_without_weights(tmp_path):
    db = tmp_path/'tasks.sqlite3'
    with TestClient(create_app(classifier=FakeClassifier(),recognition_database=db)) as c:
        headers = credential(c)
        task_id = submit(c,headers).json()['taskId']
        done = wait_result(c,headers,task_id)
        store = c.app.state.recognition.store
        with store.connect() as connection:
            row = connection.execute('SELECT * FROM recognition_jobs').fetchone()
            assert row['pixels'] is None and row['weights_json'] is None
            assert headers['Authorization'][7:] not in db.read_bytes().decode('latin1')
    with TestClient(create_app(recognition_database=db)) as c:
        assert c.get('/api/recognition/tasks/'+task_id,headers=headers).json() == done
        cap = c.get('/api/capabilities').json()['classifier']
        assert cap['historyAvailable'] and not cap['available']
        assert submit(c,headers).status_code == 503


def test_auth_expiry_and_validation(client):
    assert submit(client,{}).status_code == 401
    assert client.get('/api/recognition/tasks?token=ignored').status_code == 401
    assert client.get('/api/recognition/tasks',headers={'Authorization':'Bearer '+'a'*43}).status_code == 401
    headers = credential(client)
    assert client.get('/api/recognition/tasks?pageSize=101',headers=headers).status_code == 422
    with client.app.state.recognition.store.connect() as db:
        db.execute("UPDATE recognition_sessions SET expires_at='2000-01-01'")
    assert client.get('/api/recognition/tasks',headers=headers).status_code == 401


@pytest.mark.parametrize('roi', ['bad','{}','[1,2,3]','[0,0,999,999]','[0,0,0,10]','[false,0,20,20]','[0,0,NaN,10]'])
def test_invalid_roi_is_rejected(client,roi):
    response = submit(client,credential(client),{'roi':roi})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'INVALID_ROI'
    assert client.app.state.recognition.store.list('invalid',1,20)['total'] == 0


@pytest.mark.parametrize(('payload','code','status'), [(b'', 'EMPTY_IMAGE',422), (b'garbage','INVALID_IMAGE',422),
    (b'x'*(8*1024*1024+1),'UPLOAD_TOO_LARGE',413)], ids=['empty','corrupt','over-8mib'])
def test_invalid_upload(client,payload,code,status):
    response = submit(client,credential(client),payload=payload)
    assert response.status_code == status and response.json()['detail']['code'] == code


def test_form_and_oversized_dimensions(client):
    headers = credential(client)
    assert client.post('/api/landslide/classify',headers=headers).json()['detail']['code'] == 'IMAGE_REQUIRED'
    assert submit(client,headers,payload=png((8193,1))).json()['detail']['code'] == 'IMAGE_DIMENSIONS_EXCEEDED'
    assert submit(client,headers,{'unexpected':'value'}).status_code == 422
    stream = BytesIO()
    Image.new('RGB',(10,10)).save(stream,format='GIF')
    assert submit(client,headers,payload=stream.getvalue()).status_code == 415


def test_preprocessing_aspect_roi_and_exif_scrubbing():
    image = Image.new('RGB',(448,224),(50,50,50))
    image.info['comment'] = b'private note'
    prepared = prepare_image(image)
    assert prepared.metadata['resizedSize'] == [224,112]
    assert prepared.metadata['padding'] == [0,56]
    assert prepared.metadata['modelRoi'] == [0,56,224,168]
    assert prepared.weights[:64] == [0]*64 and prepared.weights[-64:] == [0]*64
    assert prepared.weights[64:192] == pytest.approx([1/128]*128)
    assert set(prepared.pixels[:56*224]) == {0}
    assert set(prepared.pixels[56*224:168*224]) == {50}
    preview = Image.open(BytesIO(prepared.preview))
    assert not preview.getexif() and 'comment' not in preview.info
    focused = prepare_image(Image.new('RGB',(224,224)), [0,0,14,14])
    assert focused.weights == [1]+[0]*255


def test_queue_busy_restart_and_exclusive_worker(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Blocking(FakeClassifier):
        def predict(self,p):
            entered.set()
            release.wait(5)
            return super().predict(p)
    db = tmp_path/'tasks.sqlite3'
    with TestClient(create_app(classifier=Blocking(),recognition_database=db)) as c:
        try:
            h = credential(c)
            task_id = submit(c,h).json()['taskId']
            assert entered.wait(2)
            assert c.delete('/api/recognition/tasks/'+task_id,headers=h).status_code == 409
            for _ in range(7):
                assert submit(c,h).status_code == 202
            assert submit(c,h).status_code == 429
            assert c.delete('/api/recognition/tasks/'+task_id,headers=credential(c)).status_code == 404
            with pytest.raises(RuntimeError,match='one API process'):
                WorkerLock(db).acquire()
        finally:
            release.set()
    store = TaskStore(db)
    owner_hash = store.authenticate(store.issue_session()['token'])
    pending = store.enqueue(owner_hash,prepare_image(Image.new('RGB',(20,20))),FakeClassifier.version)
    store.claim()
    with TestClient(create_app(recognition_database=db)):
        assert store.get(owner_hash,pending)['status'] == 'failed'
        assert store.get(owner_hash,pending)['errorCode'] == 'PROCESS_RESTARTED'


def test_model_failure_is_durable_without_exception_details(tmp_path):
    class Broken(FakeClassifier):
        def predict(self,p):
            raise RuntimeError('private local model path')
    with TestClient(create_app(classifier=Broken(),recognition_database=tmp_path/'tasks.db')) as c:
        h = credential(c)
        task = wait_result(c,h,submit(c,h).json()['taskId'])
        assert task['status'] == 'failed' and task['errorCode'] == 'INFERENCE_FAILED'
        assert task['result'] is None and 'private local' not in json.dumps(task)


@pytest.mark.parametrize('scores', [dict(SL=1),dict(SL=float('nan'),SP=0,FL=0,CL=1),
                                    dict(SL=1,SP=1,FL=0,CL=0),dict(SL=-.1,SP=.1,FL=0,CL=1)])
def test_invalid_model_output_cannot_become_result(scores):
    with pytest.raises(ValueError):
        classification_result(scores,FakeClassifier())


def test_manifest_rejects_modified_assets(tmp_path):
    assets = dict(baseConfig='config.json',baseWeights='model.safetensors',adapter='model.pt',normalizer='norm.npz')
    for name in assets.values():
        (tmp_path/name).write_bytes(b'synthetic, never loaded')
    spec = dict(schemaVersion=1,architecture='dinov2-base-qv-last2-r4',modelVersion='test',classes=list(CLASSES),
                preprocessingVersion=PREPROCESSING_VERSION,
                files={k:dict(path=v,sha256=sha256(tmp_path/v)) for k,v in assets.items()})
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(spec))
    assert load_manifest(path)[0]['modelVersion'] == 'test'
    (tmp_path/'norm.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='normalizer'):
        load_manifest(path)


def test_chunked_limit_applies_before_multipart_parser():
    import asyncio
    async def scenario():
        called = False
        sent = []
        async def app(scope,receive,send):
            nonlocal called
            called = True
        boundary = RecognitionBoundary(app)
        boundary.LIMIT = 10
        chunks = iter([{'type':'http.request','body':b'123456','more_body':True},
                       {'type':'http.request','body':b'78901','more_body':False}])
        async def receive():
            return next(chunks)
        async def send(message):
            sent.append(message)
        await boundary({'type':'http','path':'/api/landslide/classify','method':'POST'},receive,send)
        assert not called and sent[0]['status'] == 413
        assert (b'cache-control',b'no-store') in sent[0]['headers']
    asyncio.run(scenario())


def test_cors_allows_bearer_and_delete(client):
    response = client.options('/api/recognition/tasks/test',headers={
        'Origin':'http://localhost:5173','Access-Control-Request-Method':'DELETE',
        'Access-Control-Request-Headers':'Authorization'})
    assert response.status_code == 200


def test_configuration_requires_distinct_persistent_database(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='MARS_RECOGNITION_DB'):
        with TestClient(create_app(classifier=FakeClassifier())):
            pass
    monkeypatch.setenv('MARS_CLASSIFIER_MANIFEST',str(tmp_path/'missing.json'))
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(recognition_database=tmp_path/'tasks.db')):
            pass
