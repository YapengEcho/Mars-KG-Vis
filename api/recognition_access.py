"""Shared device authentication and request-scoped, read-only task access.

The context is an internal Python capability, never a request/response model.
It retains no raw bearer token and grants access to exactly one selected task.
"""
import re
import sqlite3
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from api.recognition_store import TaskStore


TaskId = Annotated[str, Field(strict=True, pattern=r'^[a-f0-9]{32}$', min_length=32, max_length=32)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ScreenScore = Annotated[float, Field(allow_inf_nan=False)]


class RecognitionAccessError(Exception):
    """Only fixed, public error information may cross the HTTP/tool boundary."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(code)
        self.code, self.status, self.message = code, status, message


def auth_required():
    return RecognitionAccessError('AUTH_REQUIRED', 401, 'A valid device token is required.')


def task_not_found():
    return RecognitionAccessError('TASK_NOT_FOUND', 404, 'Task was not found.')


def unavailable():
    return RecognitionAccessError('RECOGNITION_UNAVAILABLE', 503, 'Recognition storage is unavailable.')


def authenticate_owner(store: TaskStore | None, token: str | None) -> str:
    # Check missing/malformed credentials before touching unavailable storage.
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{40,128}', token):
        raise auth_required()
    if store is None:
        raise RecognitionAccessError('RECOGNITION_NOT_CONFIGURED', 503,
                                     'Recognition storage is not configured.')
    try:
        identity = store.authenticate(token)
    except (sqlite3.Error, OSError):
        raise unavailable() from None
    if identity is None:
        raise auth_required()
    return identity


def get_owned_task(store: TaskStore, identity: str, task_id: str) -> dict:
    try:
        task = store.get(identity, task_id)
    except (sqlite3.Error, OSError, ValueError):
        raise unavailable() from None
    if task is None:
        # A foreign task and an absent task have the same status and response.
        raise task_not_found()
    return task


class ResultSummary(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)
    prediction: Literal['SL', 'SP', 'FL', 'CL']
    probabilities: dict[Literal['SL', 'SP', 'FL', 'CL'], Score]
    confidence: Score
    uncertain: bool
    reviewReasons: list[str]
    reviewPolicy: str
    scoresCalibrated: bool
    modelVersion: str
    preprocessingVersion: str

    @model_validator(mode='after')
    def four_scores(self):
        if set(self.probabilities) != {'SL', 'SP', 'FL', 'CL'}:
            raise ValueError('Four class scores are required.')
        if abs(sum(self.probabilities.values()) - 1) > 1e-5:
            raise ValueError('Class scores must sum to one.')
        return self


class DetectionCandidate(BaseModel):
    """One candidate box plus its four-class classification."""
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)
    candidateId: str
    box: tuple[int, int, int, int]
    detectionScore: Score
    classification: ResultSummary


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)
    outcome: Literal['detections', 'no_detections']
    coordinateSpace: Literal['image_pixels']
    truncated: bool
    candidates: list[DetectionCandidate]

    @model_validator(mode='after')
    def outcome_matches_candidates(self):
        # 无候选时 outcome 必须是 no_detections，有候选时必须是 detections。
        if (self.outcome == 'detections') != bool(self.candidates):
            raise ValueError('Detection outcome and candidates are inconsistent.')
        return self


class ScreeningSummary(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    decision: Literal['possible_landslide', 'no_landslide', 'unsupported_image', 'inconclusive']
    score: ScreenScore | None
    threshold: ScreenScore | None
    modelVersion: str | None


class TaskSummary(BaseModel):
    """Explicit allowlist: excludes images, input metadata and model digests."""
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)
    taskId: TaskId
    kind: Literal['classify']
    status: Literal['queued', 'running', 'completed', 'failed']
    createdAt: str
    finishedAt: str | None
    modelVersion: str
    preprocessingVersion: str
    errorCode: str | None
    result: ResultSummary | None
    screening: ScreeningSummary | None = None

    @model_validator(mode='after')
    def consistent_result(self):
        expects_result = self.status == 'completed' and (
            self.screening is None or self.screening.decision == 'possible_landslide'
        )
        if expects_result != (self.result is not None) or (self.screening and self.status != 'completed'):
            raise ValueError('Task state and result are inconsistent.')
        if self.result and (self.result.modelVersion != self.modelVersion or
                            self.result.preprocessingVersion != self.preprocessingVersion):
            raise ValueError('Task and result versions differ.')
        return self


class DetectTaskSummary(BaseModel):
    """Allowlist for detection tasks: no image, no input geometry, no model digests."""
    model_config = ConfigDict(extra='ignore', frozen=True, hide_input_in_errors=True)
    taskId: TaskId
    kind: Literal['detect']
    status: Literal['queued', 'running', 'completed', 'failed']
    createdAt: str
    finishedAt: str | None
    modelVersion: str
    preprocessingVersion: str
    errorCode: str | None
    result: DetectionResult | None

    @model_validator(mode='after')
    def consistent_result(self):
        # 检测任务的 result 只在 completed 时存在，未检出也是有 result 的 completed。
        if (self.status == 'completed') != (self.result is not None):
            raise ValueError('Task state and result are inconsistent.')
        return self


class RecognitionTaskContext:
    __slots__ = ('__store', '__identity', '__task_id')

    def __init__(self, store: TaskStore, identity: str, task_id: str):
        self.__store, self.__identity, self.__task_id = store, identity, task_id

    def __repr__(self):
        return '<RecognitionTaskContext>'

    def __reduce_ex__(self, protocol):
        raise TypeError('Request authorization context cannot be serialized.')

    @property
    def task_id(self) -> str:
        return self.__task_id

    def read(self, task_id: str) -> TaskSummary | DetectTaskSummary:
        if not isinstance(task_id, str) or task_id != self.__task_id:
            raise task_not_found()
        # Read afresh: deletion or failure after preflight must not serve a cache.
        task = get_owned_task(self.__store, self.__identity, task_id)
        model = DetectTaskSummary if task.get('kind') == 'detect' else TaskSummary
        try:
            return model.model_validate(task)
        except ValidationError:
            raise unavailable() from None


def bind_task_context(store: TaskStore | None, token: str | None, task_id: str) -> RecognitionTaskContext:
    if not isinstance(task_id, str) or not re.fullmatch(r'[a-f0-9]{32}', task_id):
        raise RecognitionAccessError('INVALID_TASK_ID', 422, 'A valid task ID is required.')
    identity = authenticate_owner(store, token)
    context = RecognitionTaskContext(store, identity, task_id)
    context.read(task_id)  # Ownership/availability must pass before opening SSE.
    return context
