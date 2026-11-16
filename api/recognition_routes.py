"""Authenticated, device-scoped classification tasks and history."""
import json
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse

from api.classifier import prepare_image
from api.image_policy import MAX_UPLOAD_BYTES, ImageInputError, decode_image
from api.recognition_store import QueueFull

router = APIRouter(tags=['recognition'])
bearer = HTTPBearer(auto_error=False)


class Result(BaseModel):
    prediction: Literal['SL', 'SP', 'FL', 'CL']
    probabilities: dict[Literal['SL', 'SP', 'FL', 'CL'], float]
    confidence: float
    uncertain: bool
    reviewReasons: list[str]
    reviewPolicy: str
    scoresCalibrated: bool
    modelVersion: str
    modelDigest: str
    preprocessingVersion: str


class InputMetadata(BaseModel):
    width: int
    height: int
    roi: list[float]
    roiSource: Literal['provided_box', 'whole_image_crop']
    modelSize: list[int]
    resizedSize: list[int]
    padding: list[int]
    modelRoi: list[float]
    preprocessingVersion: str


class Task(BaseModel):
    taskId: str
    kind: Literal['classify']
    status: Literal['queued', 'running', 'completed', 'failed']
    createdAt: str
    finishedAt: str | None
    modelVersion: str
    preprocessingVersion: str
    input: InputMetadata
    result: Result | None
    errorCode: str | None
    imageUrl: str


class TaskPage(BaseModel):
    items: list[Task]
    total: int
    page: int
    pageSize: int


class Session(BaseModel):
    token: str
    tokenType: Literal['Bearer']
    expiresAt: str


def fail(code, status, message):
    raise HTTPException(status, detail={'code': code, 'message': message})


def get_worker(request: Request):
    worker = request.app.state.recognition
    if worker is None:
        fail('RECOGNITION_NOT_CONFIGURED', 503, 'Recognition storage is not configured.')
    return worker


def owner(request: Request, credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
    worker = get_worker(request)
    identity = worker.store.authenticate(credential.credentials if credential else None)
    if identity is None:
        raise HTTPException(401, detail={'code': 'AUTH_REQUIRED', 'message': 'A valid device token is required.'},
                            headers={'WWW-Authenticate': 'Bearer'})
    return identity


@router.post('/api/recognition/sessions', response_model=Session, status_code=201)
def session(request: Request):
    return get_worker(request).store.issue_session()


@router.post('/api/landslide/classify', response_model=Task, status_code=202, openapi_extra={
    'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
        'type': 'object', 'required': ['file'], 'properties': {
            'file': {'type': 'string', 'format': 'binary'},
            'roi': {'type': 'string', 'description': 'Optional JSON [x0,y0,x1,y1] in EXIF-oriented pixels.'}
        }}}}}})
async def classify(request: Request, credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
    worker = request.app.state.recognition
    if worker is None or worker.model is None:
        fail('MODEL_NOT_CONFIGURED', 503, 'Classification model is not configured.')
    if not worker.available:
        fail('WORKER_UNAVAILABLE', 503, 'Classification worker is unavailable.')
    identity = await anyio.to_thread.run_sync(worker.store.authenticate, credential.credentials if credential else None)
    if identity is None:
        raise HTTPException(401, detail={'code': 'AUTH_REQUIRED', 'message': 'A valid device token is required.'},
                            headers={'WWW-Authenticate': 'Bearer'})
    async with request.form(max_files=1, max_fields=1) as form:
        if any(key not in ('file', 'roi') for key in form):
            fail('INVALID_FORM', 422, 'Only file and roi fields are accepted.')
        upload = form.get('file')
        if not isinstance(upload, UploadFile):
            fail('IMAGE_REQUIRED', 422, 'Provide one image in the file field.')
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
        raw_roi = form.get('roi')
        try:
            roi = json.loads(raw_roi) if raw_roi is not None else None
        except (ValueError, TypeError):
            fail('INVALID_ROI', 422, 'ROI must be a JSON array of four pixel coordinates.')

    def prepare():
        return prepare_image(decode_image(payload), roi)
    try:
        prepared = await anyio.to_thread.run_sync(prepare)
    except ImageInputError as error:
        fail(error.code, error.status, str(error))
    except ValueError as error:
        fail('INVALID_ROI', 422, str(error))
    try:
        task_id = await anyio.to_thread.run_sync(worker.store.enqueue, identity, prepared, worker.model.version)
    except QueueFull:
        raise HTTPException(429, detail={'code': 'QUEUE_FULL', 'message': 'Try again after pending tasks finish.'},
                            headers={'Retry-After': '5'})
    worker.wake.set()
    task = await anyio.to_thread.run_sync(worker.store.get, identity, task_id)
    return JSONResponse(task, status_code=202, headers={'Location': f'/api/recognition/tasks/{task_id}'})


@router.get('/api/recognition/tasks', response_model=TaskPage)
def tasks(request: Request, identity=Depends(owner), page: int = Query(1, ge=1), pageSize: int = Query(20, ge=1, le=100)):
    return get_worker(request).store.list(identity, page, pageSize)


@router.get('/api/recognition/tasks/{task_id}', response_model=Task)
def task(task_id: str, request: Request, identity=Depends(owner)):
    found = get_worker(request).store.get(identity, task_id)
    if found is None:
        fail('TASK_NOT_FOUND', 404, 'Task was not found.')
    return found


@router.get('/api/recognition/tasks/{task_id}/image', response_class=Response,
            responses={200: {'content': {'image/jpeg': {}}}})
def task_image(task_id: str, request: Request, identity=Depends(owner)):
    preview = get_worker(request).store.image(identity, task_id)
    if preview is None:
        fail('TASK_NOT_FOUND', 404, 'Task was not found.')
    return Response(preview, media_type='image/jpeg', headers={'X-Content-Type-Options': 'nosniff'})


@router.delete('/api/recognition/tasks/{task_id}', status_code=204)
def delete_task(task_id: str, request: Request, identity=Depends(owner)):
    status = get_worker(request).store.delete(identity, task_id)
    if status == 'missing':
        fail('TASK_NOT_FOUND', 404, 'Task was not found.')
    if status == 'busy':
        fail('TASK_BUSY', 409, 'Wait for the running or queued task to finish before deleting it.')
    return Response(status_code=204)


class RecognitionBoundary:
    """Bound the streamed multipart body before parsing, including chunked uploads."""
    LIMIT = MAX_UPLOAD_BYTES + 64 * 1024

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        recognition = path == '/api/landslide/classify' or path.startswith('/api/recognition/')
        if scope['type'] != 'http' or not recognition:
            return await self.app(scope, receive, send)

        async def private_send(message):
            if message['type'] == 'http.response.start':
                headers = [(k,v) for k,v in message.get('headers', []) if k.lower() != b'cache-control']
                message = {**message, 'headers': headers + [(b'cache-control', b'no-store')]}
            await send(message)

        if path == '/api/landslide/classify' and scope['method'] == 'POST':
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                chunk = message.get('body', b'')
                if len(body) + len(chunk) > self.LIMIT:
                    return await JSONResponse({'detail': {'code': 'UPLOAD_TOO_LARGE',
                        'message': 'Request exceeds the upload limit.'}}, status_code=413)(scope, receive, private_send)
                body.extend(chunk)
                if not message.get('more_body', False):
                    break
            consumed = False
            async def bounded_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await receive()
            return await self.app(scope, bounded_receive, private_send)
        return await self.app(scope, receive, private_send)
