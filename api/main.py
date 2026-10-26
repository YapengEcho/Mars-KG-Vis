"""Read-only catalog service for the mobile client."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    Capabilities,
    Capability,
    Catalog,
    Health,
    Slide,
    SlidePage,
    SlideType,
    Statistics,
)


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "examples" / "catalog.demo.json"
CLASSIFICATION_REASON = "Classification model is not configured."
DETECTION_REASON = "Detection service is not configured."


def load_catalog(path: Path) -> Catalog:
    """Reject invalid catalogs at startup instead of serving partial records."""
    return Catalog.model_validate_json(path.read_text(encoding="utf-8-sig"))


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def filter_records(
    catalog: Catalog, q: str | None = None, slide_type: SlideType | None = None
) -> list[Slide]:
    records = catalog.records
    if slide_type:
        records = [record for record in records if record.type == slide_type]
    term = (q or "").strip().casefold()
    if term:
        records = [
            record
            for record in records
            if term
            in " ".join(
                (
                    record.id,
                    str(record.sourceId) if record.sourceId is not None else "",
                    record.name,
                    record.summary,
                )
            ).casefold()
        ]
    return records


def create_app(catalog_path: str | Path | None = None) -> FastAPI:
    selected_path = Path(
        catalog_path or os.environ.get("MARS_CATALOG_PATH") or DEFAULT_CATALOG
    ).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.catalog = load_catalog(selected_path)
        yield

    application = FastAPI(
        title="Mars API",
        version="0.1.0",
        description="Mars catalog browsing and service capabilities.",
        lifespan=lifespan,
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
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
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
        page: int = Query(default=1, ge=1),
        pageSize: int = Query(default=20, ge=1, le=100),
    ):
        catalog = get_catalog(request)
        records = filter_records(catalog, q, type)
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
    ):
        catalog = get_catalog(request)
        records = filter_records(catalog, q, type)
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
        return Capabilities(
            dataMode=catalog.mode,
            version=catalog.version,
            classifier=Capability(available=False, reason=CLASSIFICATION_REASON),
            detection=Capability(available=False, reason=DETECTION_REASON),
        )

    @application.post("/api/landslide/classify", status_code=503)
    def classify():
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_NOT_CONFIGURED", "message": CLASSIFICATION_REASON},
        )

    return application


app = create_app()
