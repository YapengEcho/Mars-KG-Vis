import json

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from api.main import create_app, load_catalog


@pytest.fixture
def catalog_file(tmp_path):
    records = []
    for index, kind in enumerate(("SL", "SP", "FL", "CL"), start=1):
        records.append(
            {
                "id": f"demo-{index:03d}",
                "sourceId": None,
                "name": f"Demo {kind}",
                "type": kind,
                "longitude": -170.5 + index,
                "latitude": -30 + index,
                "areaKm2": None,
                "summary": "Synthetic record",
                "imageUrl": None,
                "source": {"title": "Synthetic fixture", "url": None},
                "isDemo": True,
            }
        )
    catalog = {
        "version": "fixture-v1",
        "mode": "demo",
        "coordinateSystem": {
            "body": "Mars",
            "longitude": "east-positive, -180..180",
            "latitude": "planetocentric, degrees",
        },
        "records": records,
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


@pytest.fixture
def client(catalog_file):
    with TestClient(create_app(catalog_file)) as test_client:
        yield test_client


def test_query_pagination_and_matching_statistics(client):
    page = client.get("/api/landslides", params={"page": 2, "pageSize": 2}).json()
    assert page["total"] == 4
    assert [record["id"] for record in page["items"]] == ["demo-003", "demo-004"]
    params = {"q": "demo", "type": "SP"}
    result = client.get("/api/landslides", params=params).json()
    stats = client.get("/api/stats", params=params).json()
    assert result["total"] == stats["total"] == 1
    assert stats["byType"] == {"SL": 0, "SP": 1, "FL": 0, "CL": 0}
    assert result["items"][0]["type"] == "SP"
    assert client.get("/api/landslides", params={"q": "not-present"}).json()["total"] == 0


def test_complete_catalog_preserves_contract_and_all_records(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["mode"] == "demo"
    assert catalog["version"] == "fixture-v1"
    assert catalog["coordinateSystem"] == {
        "body": "Mars",
        "longitude": "east-positive, -180..180",
        "latitude": "planetocentric, degrees",
    }
    assert len(catalog["records"]) == 4
    assert catalog["records"][0] == client.get("/api/landslides/demo-001").json()


def test_detail_preserves_mars_coordinates_and_unknown_values(client):
    response = client.get("/api/landslides/demo-001")
    assert response.status_code == 200
    record = response.json()
    assert (record["longitude"], record["latitude"]) == (-169.5, -29)
    assert record["areaKm2"] is None
    assert record["imageUrl"] is None
    assert record["isDemo"] is True
    missing = client.get("/api/landslides/absent")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize(
    "params", [{"page": 0}, {"pageSize": 101}, {"type": "unknown"}]
)
def test_invalid_query_is_rejected(client, params):
    assert client.get("/api/landslides", params=params).status_code == 422


def test_unavailable_models_are_reported_without_prediction(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["dataMode"] == "demo"
    assert health["version"] == "fixture-v1"
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["classifier"]["available"] is False
    assert capabilities["detection"]["available"] is False
    result = client.post("/api/landslide/classify")
    assert result.status_code == 503
    assert result.json()["detail"]["code"] == "MODEL_NOT_CONFIGURED"
    assert "prediction" not in result.json()


def test_cors_allows_configured_client_and_rejects_other_origins(client):
    allowed = client.options(
        "/api/landslides",
        headers={
            "Origin": "http://localhost:4173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:4173"
    denied = client.options(
        "/api/landslides",
        headers={
            "Origin": "https://unlisted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize("invalid_case", ["duplicate", "coordinate", "mode"])
def test_invalid_catalog_is_rejected(catalog_file, invalid_case):
    value = json.loads(catalog_file.read_text(encoding="utf-8"))
    if invalid_case == "duplicate":
        value["records"][1]["id"] = value["records"][0]["id"]
    elif invalid_case == "coordinate":
        value["records"][0]["latitude"] = 100
    else:
        value["mode"] = "catalog"
    catalog_file.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_catalog(catalog_file)


def test_catalog_environment_selection(monkeypatch, catalog_file):
    monkeypatch.setenv("MARS_CATALOG_PATH", str(catalog_file))
    with TestClient(create_app()) as client:
        assert client.get("/api/landslides").json()["total"] == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceId", -1),
        ("sourceId", True),
        ("sourceId", "12"),
        ("sourceId", 1.5),
        ("source.url", "http://example.com/source"),
        ("imageUrl", "//example.com/image.png"),
        ("imageUrl", "http://example.com/image.png"),
        ("imageUrl", "images/example.png"),
    ],
)
def test_client_incompatible_record_fields_are_rejected(catalog_file, field, value):
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    record = catalog["records"][0]
    if field == "source.url":
        record["source"]["url"] = value
    else:
        record[field] = value
    catalog_file.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_catalog(catalog_file)


def test_https_and_same_origin_assets_and_nonnegative_source_id(catalog_file):
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    catalog["records"][0]["sourceId"] = 0
    catalog["records"][0]["source"]["url"] = "https://example.com/source"
    catalog["records"][0]["imageUrl"] = "/images/example.png"
    catalog["records"][1]["imageUrl"] = "https://example.com/image.png"
    catalog_file.write_text(json.dumps(catalog), encoding="utf-8")
    validated = load_catalog(catalog_file)
    assert validated.records[0].sourceId == 0
    assert validated.records[0].imageUrl == "/images/example.png"
    assert validated.records[1].imageUrl == "https://example.com/image.png"
