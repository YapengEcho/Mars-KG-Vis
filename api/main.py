"""Read-only catalog service for the mobile client."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.sqlite_store import load_sqlite_catalog
from api.catalog_query import Bounds, filter_records, parse_bbox
from api.content_routes import router as content_router
from api.classifier import DinoClassifier, PREPROCESSING_VERSION, REVIEW_POLICY
from api.recognition_routes import RecognitionBoundary, router as recognition_router
from api.recognition_store import TaskStore
from api.recognition_worker import RecognitionWorker

from api.agent import llm
from api.agent.records import TraceRecorder
from api.agent.retrieval import KeywordRetriever, load_docs
from api.agent.schemas import ChatRequest, ChatResponse
from api.agent.service import iter_agent_events, run_agent
from api.schemas import (
    AgentCapability,
    Capabilities,
    Capability,
    ClassifierCapability,
    Catalog,
    Health,
    Slide,
    SlidePage,
    SlideType,
    Statistics,
)


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "examples" / "catalog.demo.json"
DEFAULT_DOCS = Path(__file__).resolve().parent / "data" / "docs" / "guidebook.demo.json"
CLASSIFICATION_REASON = "Classification model is not configured."
DETECTION_REASON = "Detection service is not configured."
LLM_REASON = "LLM API key is not configured."


def build_retriever() -> KeywordRetriever:
    docs_path = os.environ.get("MARS_DOCS_PATH")
    if docs_path:
        return KeywordRetriever(load_docs(docs_path))
    if DEFAULT_DOCS.exists():
        return KeywordRetriever(load_docs(DEFAULT_DOCS))
    return KeywordRetriever([])


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
               asset_root: str | Path | None = None, classifier=None,
               recognition_database: str | Path | None = None) -> FastAPI:
    if database_path is not None and catalog_path is not None:
        raise ValueError("Choose a catalog file or a database, not both")
    selected_database = database_path or (os.environ.get("MARS_DB_PATH") if catalog_path is None else None)
    selected_path = Path(
        catalog_path or os.environ.get("MARS_CATALOG_PATH") or DEFAULT_CATALOG
    ).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.catalog = (
            load_sqlite_catalog(Path(selected_database))
            if selected_database else load_catalog(selected_path)
        )
        application.state.retriever = build_retriever()
        application.state.tracer = TraceRecorder()
        application.state.content_database = Path(selected_database).resolve() if selected_database else None
        configured_root = asset_root or os.environ.get('MARS_ASSET_ROOT')
        application.state.asset_root = Path(configured_root).expanduser().resolve() if configured_root else None
        model = classifier
        manifest = os.environ.get('MARS_CLASSIFIER_MANIFEST')
        if model is None and manifest:
            model = DinoClassifier(manifest, device=os.environ.get('MARS_CLASSIFIER_DEVICE', 'cpu'),
                                   threads=int(os.environ.get('MARS_CLASSIFIER_THREADS', '2')))
        task_database = recognition_database or os.environ.get('MARS_RECOGNITION_DB')
        if model is not None and not task_database:
            raise ValueError('Set MARS_RECOGNITION_DB for durable recognition records.')
        if task_database and selected_database and Path(task_database).resolve() == Path(selected_database).resolve():
            raise ValueError('Recognition records must use a database separate from the geological catalog.')
        worker = RecognitionWorker(TaskStore(task_database), model) if task_database else None
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

    @application.get("/api/catalog", response_model=Catalog)
    def complete_catalog(request: Request):
        return get_catalog(request)

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
        return Capabilities(
            dataMode=catalog.mode,
            version=catalog.version,
            classifier=ClassifierCapability(available=ready,
                reason=('Ready; results require review.' if ready else
                        'Classification worker is unavailable.' if model else CLASSIFICATION_REASON),
                modelVersion=model.version if model else None,
                preprocessingVersion=PREPROCESSING_VERSION if model else None,
                reviewPolicy=REVIEW_POLICY if model else None,
                historyAvailable=worker is not None),
            detection=Capability(available=False, reason=DETECTION_REASON),
            agent=AgentCapability(
                available=agent_available,
                model=llm.model_name(),
                reason=None if agent_available else LLM_REASON,
            ),
        )

    @application.post("/api/agent/chat", response_model=ChatResponse)
    def agent_chat(payload: ChatRequest, request: Request):
        return run_agent(
            payload,
            get_catalog(request),
            request.app.state.retriever,
            request.app.state.tracer,
        )

    @application.post("/api/agent/chat/stream")
    def agent_chat_stream(payload: ChatRequest, request: Request):
        """Server-sent events variant of /api/agent/chat.

        Emits one ``data:`` frame per agent event (delta / status / done /
        error) so clients can render the answer incrementally.
        """

        def event_source():
            for event in iter_agent_events(
                payload,
                get_catalog(request),
                request.app.state.retriever,
                request.app.state.tracer,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


app = create_app()
