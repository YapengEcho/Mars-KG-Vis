import csv
import json

import pytest

import download


@pytest.fixture
def catalog_site(monkeypatch):
    row = (
        '{id:493,name:"滑坡 493",type:"SL",area:2.5,Perimeter:7.5,'
        'lon_degree:"-52",lat_degree:"6",lon:-52,lat:6,desc:"样例"}'
    )
    catalog = "const records=[" + row + "," + row.replace("493", "597") + "];"
    pages = {
        "/": '<script type="module" src="/assets/index-Ab3.js"></script>',
        "/assets/index-Ab3.js": 'const load=()=>import("./MarsViewer-Cd4.js");',
        "/assets/MarsViewer-Cd4.js": 'import{rows}from"./project-catalog-Ef5.js";',
        "/assets/project-catalog-Ef5.js": catalog,
    }
    visited = []

    def get(url, *, label=None):
        assert url.startswith(download.BASE_URL + "/")
        path = url.removeprefix(download.BASE_URL)
        visited.append(path)
        return pages[path]

    monkeypatch.setattr(download, "http_get", get)
    return pages, visited


def test_catalog_follows_map_script_import(catalog_site):
    pages, visited = catalog_site

    assert [row["source_id"] for row in download.fetch_catalog_records()] == [493, 597]
    assert visited == list(pages)


def test_legacy_catalog_is_in_map_script(catalog_site):
    pages, visited = catalog_site
    pages["/assets/MarsViewer-Cd4.js"] = pages.pop("/assets/project-catalog-Ef5.js")

    assert [row["source_id"] for row in download.fetch_catalog_records()] == [493, 597]
    assert visited == list(pages)


def test_single_id_export_writes_only_requested_record(catalog_site, tmp_path):
    download.main(["--source-id", "493", "--output-dir", str(tmp_path)])

    assert {path.name for path in tmp_path.iterdir()} == {
        "mars_landslide_0493.csv", "mars_landslide_0493.json"
    }
    payload = json.loads((tmp_path / "mars_landslide_0493.json").read_text(encoding="utf-8"))
    with (tmp_path / "mars_landslide_0493.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [record["id"] for record in payload] == ["ls_0493"]
    assert rows == [{key: str(value) for key, value in payload[0].items()}]


def test_unknown_id_does_not_create_exports(catalog_site, tmp_path):
    with pytest.raises(SystemExit) as exc:
        download.main(["--source-id", "123", "--output-dir", str(tmp_path)])

    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_default_export_preserves_local_catalog(catalog_site, monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    existing = tmp_path / "mars_landslides.csv"
    existing.write_text("local catalog", encoding="utf-8")

    download.main([])

    assert existing.read_text(encoding="utf-8") == "local catalog"
    with (tmp_path / "geomars_download" / "mars_landslides.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        assert [row["source_id"] for row in csv.DictReader(f)] == ["493", "597"]
