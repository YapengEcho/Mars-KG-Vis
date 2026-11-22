"""Read-only catalog service for the mobile client."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from api.public_catalog import load_public_catalog, PUBLIC_IMAGES
from api.sqlite_store import load_sqlite_catalog
from api.catalog_query import Bounds, filter_records, parse_bbox
from api.content_routes import router as content_router
from api.classifier import DinoClassifier, PREPROCESSING_VERSION, REVIEW_POLICY
from api.recognition_routes import RecognitionBoundary, agent_task_context, bearer, router as recognition_router
from api.recognition_access import RecognitionAccessError
from api.recognition_store import TaskStore
from api.recognition_worker import RecognitionWorker
from api.detector import (DETECTION_PREPROCESSING_VERSION, MAX_CANDIDATES, MAX_DETECTION_PIXELS,
                          MAX_DETECTION_SIDE, YoloDetector)
from api.image_policy import MAX_UPLOAD_BYTES

from api.agent import llm
from api.agent.records import TraceRecorder
from api.agent.retrieval import KeywordRetriever, load_docs
from api.agent.schemas import ChatRequest, ChatResponse, ExternalChatContext, ExternalContextRequest
from api.agent.service import external_chat_context, iter_agent_events, run_agent
from api.schemas import (
    AgentCapability,
    Capabilities,
    ClassifierCapability,
    DetectionCapability,
    Catalog,
    Health,
    Slide,
    SlidePage,
    SlideType,
    Statistics,
)


DEFAULT_DOCS = Path(__file__).resolve().parent / "data" / "docs" / "guidebook.demo.json"
CLASSIFICATION_REASON = "Classification model is not configured."
SCREENING_REASON = "The landslide screening model is not configured."
DETECTION_REASON = "Detection service is not configured."
LLM_REASON = "LLM API key is not configured."
AGENT_TASK_ERRORS = {
    401: {'description': 'A selected recognition task requires a valid device Bearer credential.'},
    404: {'description': 'The selected task does not exist or is not owned by this device.'},
    503: {'description': 'Recognition storage is not configured or is unavailable.'},
}


def build_public_retriever() -> KeywordRetriever:
    if DEFAULT_DOCS.exists():
        return KeywordRetriever(load_docs(DEFAULT_DOCS))
    return KeywordRetriever([])


def build_retriever(public_retriever: KeywordRetriever | None = None) -> KeywordRetriever:
    docs_path = os.environ.get("MARS_DOCS_PATH")
    if docs_path:
        return KeywordRetriever(load_docs(docs_path))
    return public_retriever if public_retriever is not None else build_public_retriever()


def load_catalog(path: Path) -> Catalog:
    """Reject invalid catalogs at startup instead of serving partial records."""
    return Catalog.model_validate_json(path.read_text(encoding="utf-8-sig"))


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def query_bounds(bbox: str | None = Query(default=None, max_length=160)) -> Bounds | None:
    try:
        return parse_bbox(bbox)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_BBOX", "message": str(error)}) from error


def create_app(catalog_path: str | Path | None = None, *, database_path: str | Path | None = None,
               asset_root: str | Path | None = None, classifier=None, binary_screen=None,
               recognition_database: str | Path | None = None, detector=None) -> FastAPI:
    if database_path is not None and catalog_path is not None:
        raise ValueError("Choose a catalog file or a database, not both")
    selected_database = database_path or (os.environ.get("MARS_DB_PATH") if catalog_path is None else None)
    configured_catalog = catalog_path or os.environ.get("MARS_CATALOG_PATH")
    selected_path = Path(configured_catalog).expanduser().resolve() if configured_catalog else None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if selected_database:
            application.state.catalog = load_sqlite_catalog(Path(selected_database))
        elif selected_path:
            application.state.catalog = load_catalog(selected_path)
        else:
            application.state.catalog = load_public_catalog()
        public_retriever = build_public_retriever()
        application.state.public_retriever = public_retriever
        application.state.retriever = build_retriever(public_retriever)
        application.state.tracer = TraceRecorder()
        application.state.content_database = Path(selected_database).resolve() if selected_database else None
        configured_root = asset_root or os.environ.get('MARS_ASSET_ROOT')
        application.state.asset_root = Path(configured_root).expanduser().resolve() if configured_root else None
        model = classifier
        manifest = os.environ.get('MARS_CLASSIFIER_MANIFEST')
        if model is None and manifest:
            model = DinoClassifier(manifest, device=os.environ.get('MARS_CLASSIFIER_DEVICE', 'cpu'),
                                   threads=int(os.environ.get('MARS_CLASSIFIER_THREADS', '2')))
        screen = binary_screen
        screen_manifest = os.environ.get('MARS_BINARY_SCREEN_MANIFEST')
        if model is not None and screen is None and screen_manifest:
            from api.binary_screen import BinaryScreen

            screen = BinaryScreen(screen_manifest, device=os.environ.get('MARS_CLASSIFIER_DEVICE', 'cpu'))
        task_database = recognition_database or os.environ.get('MARS_RECOGNITION_DB')
        if model is not None and not task_database:
            raise ValueError('Set MARS_RECOGNITION_DB for durable recognition records.')
        if task_database and selected_database and Path(task_database).resolve() == Path(selected_database).resolve():
            raise ValueError('Recognition records must use a database separate from the geological catalog.')
        detection_model = detector
        weights = os.environ.get('MARS_DETECTOR_WEIGHTS')
        if detection_model is None and weights:
            if model is None or not task_database:
                raise ValueError('Configured detection needs a classifier and a private recognition database.')
            detection_model = YoloDetector(weights, os.environ.get('MARS_DETECTOR_SHA256', ''),
                os.environ.get('MARS_DETECTOR_VERSION', ''), device=os.environ.get('MARS_DETECTOR_DEVICE', 'cpu'))
        if detection_model is not None and (model is None or not task_database):
            raise ValueError('Configured detection needs a classifier and a private recognition database.')
        worker = (
            RecognitionWorker(TaskStore(task_database), model, detector=detection_model, binary_screen=screen)
            if task_database else None
        )
        application.state.recognition = worker
        if worker:
            worker.start()
        try:
            yield
        finally:
            if worker:
                worker.close()

    application = FastAPI(
        title="Mars API",
        version="0.1.0",
        description="Mars catalog browsing and service capabilities.",
        lifespan=lifespan,
    )
    application.include_router(content_router)
    application.include_router(recognition_router)
    application.add_middleware(RecognitionBoundary)

    @application.exception_handler(RecognitionAccessError)
    async def recognition_access_error(request: Request, error: RecognitionAccessError):
        headers = {'Cache-Control': 'no-store'}
        if error.status == 401:
            headers['WWW-Authenticate'] = 'Bearer'
        return JSONResponse({'detail': {'code': error.code, 'message': error.message}},
                            status_code=error.status, headers=headers)

    application.mount('/samples', StaticFiles(directory=PUBLIC_IMAGES), name='public-sample-images')
    case_images = os.environ.get('MARS_LOCAL_CASE_IMAGES')
    if case_images:
        application.mount(
            '/case-images', StaticFiles(directory=Path(case_images).expanduser().resolve()),
            name='local-case-images',
        )
    default_origins = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173,"
        "http://localhost,https://localhost,capacitor://localhost"
    )
    origins = [
        origin.strip()
        for origin in os.environ.get("MARS_CORS_ORIGINS", default_origins).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @application.get("/api/health", response_model=Health)
    def health(request: Request):
        catalog = get_catalog(request)
        return Health(
            status="ok", service="mars-api", dataMode=catalog.mode, version=catalog.version
        )

    @application.get(
        "/api/catalog",
        response_model=Catalog,
        responses={
            200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
            409: {"description": "The requested catalogue version has changed."},
        },
    )
    def complete_catalog(
        request: Request,
        chart: Literal["types", "latitude"] | None = None,
        type: SlideType | None = None,
        version: str | None = None,
    ):
        catalog = get_catalog(request)
        if chart is None:
            return catalog

        from api.catalog_charts import render_catalog_chart

        if version is not None and catalog.version != version:
            raise HTTPException(status_code=409, detail="The requested catalogue version has changed.")
        return Response(
            render_catalog_chart(catalog, chart, type),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @application.get("/api/landslides", response_model=SlidePage)
    def list_landslides(
        request: Request,
        q: str | None = Query(default=None, max_length=200),
        type: SlideType | None = None,
        bounds: Bounds | None = Depends(query_bounds),
        page: int = Query(default=1, ge=1),
        pageSize: int = Query(default=20, ge=1, le=100),
    ):
        catalog = get_catalog(request)
        records = filter_records(catalog, q, type, bounds)
        start = (page - 1) * pageSize
        return SlidePage(
            items=records[start : start + pageSize],
            total=len(records),
            page=page,
            pageSize=pageSize,
            dataMode=catalog.mode,
            version=catalog.version,
        )

    @application.get("/api/landslides/{slide_id}", response_model=Slide)
    def get_landslide(slide_id: str, request: Request):
        for record in get_catalog(request).records:
            if record.id == slide_id:
                return record
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": "Landslide record was not found."},
        )

    @application.get("/api/stats", response_model=Statistics)
    def statistics(
        request: Request,
        q: str | None = Query(default=None, max_length=200),
        type: SlideType | None = None,
        bounds: Bounds | None = Depends(query_bounds),
    ):
        catalog = get_catalog(request)
        records = filter_records(catalog, q, type, bounds)
        counts = {kind: 0 for kind in ("SL", "SP", "FL", "CL")}
        for record in records:
            counts[record.type] += 1
        return Statistics(
            total=len(records),
            byType=counts,
            dataMode=catalog.mode,
            version=catalog.version,
        )

    @application.get("/api/capabilities", response_model=Capabilities)
    def capabilities(request: Request):
        catalog = get_catalog(request)
        agent_available = llm.is_configured()
        worker = request.app.state.recognition
        model = worker.model if worker else None
        ready = bool(worker and worker.available)
        screen_configured = bool(worker and worker.binary_screen is not None)
        reason = CLASSIFICATION_REASON
        if model is not None:
            if not screen_configured:
                reason = SCREENING_REASON
            else:
                reason = 'Ready; results require review.' if ready else 'Classification worker is unavailable.'
        return Capabilities(
            dataMode=catalog.mode,
            version=catalog.version,
            classifier=ClassifierCapability(available=ready,
                reason=reason,
                modelVersion=model.version if model else None,
                preprocessingVersion=PREPROCESSING_VERSION if model else None,
                reviewPolicy=REVIEW_POLICY if model else None,
                historyAvailable=worker is not None,
                screeningAvailable=screen_configured),
            detection=DetectionCapability(available=bool(worker and worker.detection_available), reason=(
                'Ready; results require review.' if worker and worker.detection_available else
                'Recognition worker is unavailable.' if worker and worker.detector is not None else DETECTION_REASON),
                modelVersion=worker.detector.version if worker and worker.detector else None,
                preprocessingVersion=DETECTION_PREPROCESSING_VERSION,
                maxImageSide=MAX_DETECTION_SIDE, maxImagePixels=MAX_DETECTION_PIXELS,
                maxUploadBytes=MAX_UPLOAD_BYTES, maxCandidates=MAX_CANDIDATES),
            agent=AgentCapability(
                available=agent_available,
                model=llm.model_name(),
                reason=None if agent_available else LLM_REASON,
            ),
        )

    @application.post("/api/agent/context", response_model=ExternalChatContext)
    def agent_context(payload: ExternalContextRequest, request: Request, response: Response):
        """Provide public context to the browser without calling a language model."""
        response.headers["Cache-Control"] = "no-store"
        return external_chat_context(
            payload.question,
            get_catalog(request),
            request.app.state.public_retriever,
            request.app.state.tracer,
        )

    @application.post("/api/agent/chat", response_model=ChatResponse,
                      responses=AGENT_TASK_ERRORS, openapi_extra={'security': [{}]})
    def agent_chat(payload: ChatRequest, request: Request,
                   credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """Public chat; selecting recognitionTaskId requires its device credential.

        Selected tasks currently return a stored summary without a language-model call.
        """
        context = agent_task_context(request, credential, payload.recognitionTaskId)
        return run_agent(
            payload,
            get_catalog(request),
            request.app.state.retriever,
            request.app.state.tracer,
            recognition_context=context,
        )

    @application.post("/api/agent/chat/stream", response_class=StreamingResponse,
                      responses={**AGENT_TASK_ERRORS, 200: {
                          'description': 'SSE status/delta/done/error frames; terminal frames contain ChatResponse.',
                          'content': {'text/event-stream': {'schema': {'type': 'string'}}},
                      }}, openapi_extra={'security': [{}]})
    def agent_chat_stream(payload: ChatRequest, request: Request,
                          credential: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """Server-sent events variant of /api/agent/chat.

        Emits one ``data:`` frame per agent event (delta / status / done /
        error) so clients can render the answer incrementally.
        A selected recognitionTaskId requires its device Bearer credential;
        authorization failures return ordinary JSON before SSE opens.
        """

        context = agent_task_context(request, credential, payload.recognitionTaskId)

        def event_source():
            for event in iter_agent_events(
                payload,
                get_catalog(request),
                request.app.state.retriever,
                request.app.state.tracer,
                recognition_context=context,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return application


app = create_app()
