PRAGMA foreign_keys = ON;

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT, notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE landslides (
    id TEXT PRIMARY KEY,
    source_id INTEGER UNIQUE CHECK(source_id IS NULL OR source_id >= 0),
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('SL','SP','FL','CL')),
    raw_type TEXT NOT NULL,
    longitude REAL NOT NULL CHECK(longitude BETWEEN -180 AND 180),
    latitude REAL NOT NULL CHECK(latitude BETWEEN -90 AND 90),
    area_km2 REAL CHECK(area_km2 IS NULL OR area_km2 >= 0),
    perimeter REAL,
    summary TEXT NOT NULL DEFAULT '',
    image_url TEXT,
    source_ref TEXT NOT NULL REFERENCES sources(id),
    is_demo INTEGER NOT NULL CHECK(is_demo IN (0,1)),
    raw_record_json TEXT NOT NULL
);
CREATE INDEX landslides_type ON landslides(type);
CREATE INDEX landslides_location ON landslides(longitude, latitude);
CREATE TABLE assets (
    id TEXT PRIMARY KEY,
    landslide_id TEXT REFERENCES landslides(id),
    kind TEXT NOT NULL CHECK(kind IN ('image','point','polygon','basemap','document')),
    path TEXT NOT NULL, mime_type TEXT, sha256 TEXT,
    width INTEGER, height INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE atlas_types (
    type TEXT PRIMARY KEY CHECK(type IN ('SL','SP','FL','CL')),
    title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('draft','ready')) DEFAULT 'draft'
);
CREATE TABLE geometry_links (
    landslide_id TEXT NOT NULL REFERENCES landslides(id),
    kind TEXT NOT NULL CHECK(kind IN ('point','polygon')),
    asset_id TEXT NOT NULL REFERENCES assets(id),
    record_index0 INTEGER NOT NULL CHECK(record_index0 >= 0),
    raw_type TEXT NOT NULL,
    PRIMARY KEY(landslide_id, kind)
);
CREATE TABLE cases (
    id TEXT PRIMARY KEY, landslide_id TEXT NOT NULL REFERENCES landslides(id),
    title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    image_asset_id TEXT REFERENCES assets(id),
    status TEXT NOT NULL CHECK(status IN ('draft','ready')) DEFAULT 'draft'
);
CREATE TABLE documents (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT, local_path TEXT, citation TEXT NOT NULL DEFAULT ''
);
CREATE TABLE case_documents (
    case_id TEXT NOT NULL REFERENCES cases(id), document_id TEXT NOT NULL REFERENCES documents(id),
    locator TEXT NOT NULL DEFAULT '', PRIMARY KEY(case_id, document_id, locator)
);
-- Task persistence is reserved for later inference integration; no task endpoint is exposed yet.
CREATE TABLE recognition_tasks (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('classify','detect')),
    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
    input_asset_id TEXT REFERENCES assets(id), model_version TEXT,
    created_at TEXT NOT NULL, finished_at TEXT, result_json TEXT, error_code TEXT
);
PRAGMA user_version = 1;
