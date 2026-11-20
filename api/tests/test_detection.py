from io import BytesIO
import sqlite3
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.detector import MAX_CANDIDATES, DetectionBox, YoloDetector, detect_and_classify, prepare_detection
from api.main import create_app
from api.recognition_store import TaskStore


class Classifier:
    version = 'synthetic-classifier'
    asset_digest = '0' * 64

    def predict(self, prepared):
        assert prepared.metadata['roiSource'] == 'provided_box' or prepared.metadata['width'] == 32
        return {'SL': .1, 'SP': .2, 'FL': .3, 'CL': .4}


class Detector:
    version = 'synthetic-detector'

    def predict(self, image):
        assert image.size == (32, 24)
        return [DetectionBox(3, 4, 15, 20, .8)]


class ScreeningModel:
    version = 'synthetic-screen'
    threshold = .5

    def predict(self, image, roi=None):
        return .8


def image(size=(32, 24)):
    stream = BytesIO()
    Image.new('RGB', size, 'gray').save(stream, format='PNG')
    return {'file': ('image.png', stream.getvalue(), 'image/png')}


def credential(client):
    return {'Authorization': 'Bearer ' + client.post('/api/recognition/sessions').json()['token']}


def wait_for_task(client, url, headers):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(url, headers=headers)
        assert response.status_code == 200
        if response.json()['status'] in ('completed', 'failed'):
            return response.json()
        time.sleep(.02)
    pytest.fail('Task did not terminate')


@pytest.fixture(autouse=True)
def local_config(monkeypatch):
    for key in ('MARS_CLASSIFIER_MANIFEST', 'MARS_BINARY_SCREEN_MANIFEST', 'MARS_DETECTOR_WEIGHTS',
                'MARS_RECOGNITION_DB', 'MARS_DB_PATH', 'MARS_CATALOG_PATH'):
        monkeypatch.delenv(key, raising=False)


def test_detection_task_isolation_history_and_box_contract(tmp_path):
    db = tmp_path / 'jobs.db'
    with TestClient(create_app(classifier=Classifier(), binary_screen=ScreeningModel(), detector=Detector(),
                               recognition_database=db)) as client:
        capability = client.get('/api/capabilities').json()['detection']
        assert capability['available'] and capability['maxImageSide'] == 2048
        a, b = credential(client), credential(client)
        assert client.post('/api/landslide/detect', files=image()).status_code == 401
        submitted = client.post('/api/landslide/detect', headers=a, files=image())
        assert submitted.status_code == 202 and submitted.headers['cache-control'] == 'no-store'
        url = submitted.headers['location']
        task = wait_for_task(client, url, a)
        assert task['kind'] == 'detect' and task['status'] == 'completed'
        assert 'screening' not in task
        assert task['modelVersion'] == Detector.version
        assert (task['input']['width'], task['input']['height']) == (32, 24)
        result = task['result']
        assert result['outcome'] == 'detections' and result['coordinateSpace'] == 'image_pixels'
        candidate, = result['candidates']
        assert candidate['box'] == [3, 4, 15, 20] and candidate['detectionScore'] == .8
        assert candidate['classification']['prediction'] == 'CL'
        assert candidate['classification']['modelVersion'] == Classifier.version
        assert candidate['classification']['uncertain'] and not candidate['classification']['scoresCalibrated']
        preview = client.get(task['imageUrl'], headers=a)
        assert preview.headers['cache-control'] == 'no-store'
        assert Image.open(BytesIO(preview.content)).size == (32, 24)
        assert client.get(url, headers=b).status_code == 404
        assert client.get(task['imageUrl'], headers=b).status_code == 404
        assert client.delete(url, headers=b).status_code == 404
        assert client.get('/api/recognition/tasks?kind=detect', headers=b).json()['items'] == []
        assert client.get('/api/recognition/tasks?kind=detect', headers=a).json()['items'] == [task]
        classification = client.post('/api/landslide/classify', headers=a, files=image())
        assert classification.status_code == 202
        classified = wait_for_task(client, classification.headers['location'], a)
        assert classified['screening']['decision'] == 'possible_landslide'
        assert client.get('/api/recognition/tasks', headers=a).json()['items'] == [classified]
    with TestClient(create_app(recognition_database=db)) as client:
        assert not client.get('/api/capabilities').json()['detection']['available']
        assert client.get(url, headers=a).json() == task
        assert client.delete(url, headers=a).status_code == 204
        assert client.get(url, headers=a).status_code == 404


def test_detection_works_without_single_image_screening(tmp_path):
    with TestClient(create_app(classifier=Classifier(), detector=Detector(),
                               recognition_database=tmp_path / 'jobs.db')) as client:
        capabilities = client.get('/api/capabilities').json()
        assert capabilities['detection']['available']
        assert not capabilities['classifier']['available']
        owner, stranger = credential(client), credential(client)
        classified = client.post('/api/landslide/classify', headers=owner, files=image())
        assert classified.status_code == 503
        assert classified.json()['detail']['code'] == 'SCREENING_NOT_CONFIGURED'
        detection = client.post('/api/landslide/detect', headers=owner, files=image())
        assert detection.status_code == 202
        body = {
            'recognitionTaskId': detection.json()['taskId'],
            'messages': [{'role': 'user', 'content': '请查看这条识别记录'}],
        }
        for endpoint in ('/api/agent/chat', '/api/agent/chat/stream'):
            response = client.post(endpoint, headers=stranger, json=body)
            assert response.status_code == 404
            assert response.json()['detail']['code'] == 'TASK_NOT_FOUND'
            response = client.post(endpoint, headers=owner, json=body)
            assert response.status_code == 422 and response.headers['cache-control'] == 'no-store'
            assert response.json()['detail']['code'] == 'UNSUPPORTED_TASK_KIND'
        assert wait_for_task(client, detection.headers['location'], owner)['status'] == 'completed'


@pytest.mark.parametrize('raise_error', [False, True])
def test_empty_detection_is_distinct_from_failed(tmp_path, raise_error):
    class Alternative(Detector):
        def predict(self, image):
            if raise_error:
                raise ValueError('internal detail')
            return []

    with TestClient(create_app(classifier=Classifier(), detector=Alternative(),
                               recognition_database=tmp_path/'jobs.db')) as client:
        auth = credential(client)
        assert client.post('/api/landslide/detect', headers=auth, files=image((2049, 1))).status_code == 413
        submitted = client.post('/api/landslide/detect', headers=auth, files=image())
        task = wait_for_task(client, submitted.headers['location'], auth)
        if raise_error:
            assert task['status'] == 'failed' and task['result'] is None
            assert task['errorCode'] == 'INFERENCE_FAILED' and 'internal detail' not in str(task)
        else:
            assert task['status'] == 'completed' and task['result']['outcome'] == 'no_detections'
            assert task['result']['candidates'] == []


def test_existing_store_schema_is_migrated_without_losing_sessions_or_tasks(tmp_path):
    path = tmp_path / 'old.db'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE recognition_sessions(token_hash TEXT PRIMARY KEY, created_at TEXT, expires_at TEXT);
            CREATE TABLE recognition_jobs(id TEXT PRIMARY KEY, owner_hash TEXT, status TEXT, created_at TEXT,
                finished_at TEXT, model_version TEXT, preprocessing_version TEXT, input_json TEXT,
                pixels BLOB, weights_json TEXT, preview BLOB, result_json TEXT, error_code TEXT);
            INSERT INTO recognition_sessions VALUES('owner', 'yesterday', 'tomorrow');
            INSERT INTO recognition_jobs(id,owner_hash,status,input_json) VALUES('old', 'owner', 'completed', '{}');
        ''')
    store = TaskStore(path)
    assert store.get('owner', 'old')['kind'] == 'classify'
    assert store.list('owner', 1, 20)['total'] == 1
    assert store.list('owner', 1, 20, 'detect')['total'] == 0


def test_tile_predictions_are_reprojected_and_deduplicated_before_classification():
    # One object visible in both windows. At width 1100 the right window starts at x=76.
    class Array(list):
        def cpu(self):
            return self

        def tolist(self):
            return list(self)

    class Model:
        def __init__(self):
            self.positions = iter((876, 800))

        def predict(self, **kwargs):
            assert kwargs['source'].size == (1024, 100)
            left = next(self.positions)
            boxes = SimpleNamespace(xyxy=Array([[left, 20, left + 60, 60]]), conf=Array([.9]))
            return [SimpleNamespace(boxes=boxes)]

    detector = object.__new__(YoloDetector)
    detector.device, detector.model = 'cpu', Model()
    field = prepare_detection(Image.new('RGB', (1100, 100)))
    result = detect_and_classify(field.pixels, detector, Classifier())
    assert [candidate['box'] for candidate in result['candidates']] == [[876, 20, 936, 60]]


def test_candidate_limit_reports_truncation():
    class Crowded(Detector):
        def predict(self, image):
            return [DetectionBox(x, 4, x + 2, 20, .8) for x in range(0, 60, 4)]

    pixels = prepare_detection(Image.new('RGB', (64, 24))).pixels
    result = detect_and_classify(pixels, Crowded(), Classifier())
    assert result['truncated'] and len(result['candidates']) == MAX_CANDIDATES
