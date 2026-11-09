"""Import curated atlas text and serve published SQLite content."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

EXTRA_SCHEMA = '''CREATE TABLE IF NOT EXISTS atlas_content (
    kind TEXT NOT NULL CHECK(kind IN ('type','case','comparison')),
    id TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY(kind,id));'''
TYPES = ('SL', 'SP', 'FL', 'CL')


@contextmanager
def read_database(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        yield db


def metadata(db):
    return {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM metadata')}


def content_payload(db, kind, identity):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='atlas_content'").fetchone():
        return None
    row = db.execute('SELECT payload_json FROM atlas_content WHERE kind=? AND id=?', (kind, identity)).fetchone()
    return json.loads(row[0]) if row else None


def import_atlas(database: Path, content_path: Path, *, publish=False):
    """Accept the atlas.json format from the content owner; preserve existing assets.

    Coordinate records are never imported from this text file. All referenced
    slides must exist and agree with the supplied type. Existing publication
    states are preserved unless publish=True explicitly promotes the content.
    """
    raw = content_path.read_bytes()
    payload = json.loads(raw.decode('utf-8-sig'))
    source = payload['sourceDocument']
    if not payload.get('version') or not source.get('id') or not source.get('title'):
        raise ValueError('Content version and source identity are required')
    if source.get('url') and not source['url'].startswith('https://'):
        raise ValueError('Source URL must use HTTPS')
    kinds = payload['types']; cases = payload['cases']; comparisons = payload.get('comparisons', [])
    if len(kinds) != 4 or {k['type'] for k in kinds} != set(TYPES):
        raise ValueError('Exactly four distinct atlas types are required')
    case_ids = [c['id'] for c in cases]
    if len(case_ids) != len(set(case_ids)) or any(not i for i in case_ids):
        raise ValueError('Case IDs must be unique and nonempty')
    registered = {}
    for kind in kinds:
        if not kind.get('name') or not kind.get('quote') or not kind.get('citations'):
            raise ValueError('Type name, definition and citations are required')
        for cid in kind.get('caseIds', []):
            if cid in registered:
                raise ValueError('Case registered more than once')
            registered[cid] = kind['type']
        for citation in kind.get('citations', []):
            if citation['documentId'] != source['id'] or not citation.get('locator'):
                raise ValueError('Unknown or unlocated type citation')
    if set(registered) != set(case_ids):
        raise ValueError('Type-to-case links do not match case list')
    comparison_ids=[i['id'] for i in comparisons]
    if len(set(comparison_ids))!=len(comparison_ids) or any(not i for i in comparison_ids):
        raise ValueError('Comparison IDs must be unique and nonempty')
    for item in comparisons:
        if item['left'] not in TYPES or item['right'] not in TYPES or item['left'] == item['right']:
            raise ValueError('Invalid comparison types')
        for citation in item.get('citations', []):
            if citation['documentId'] != source['id'] or not citation.get('locator'):
                raise ValueError('Unknown comparison citation')
    database = database.resolve()
    # mode=rw prevents an accidental empty database when a path is mistyped.
    with sqlite3.connect(database.as_uri()+'?mode=rw', uri=True) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('BEGIN IMMEDIATE')
        try:
            if db.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('Unsupported schema')
            db.execute(EXTRA_SCHEMA)
            for case in cases:
                row = db.execute('SELECT type FROM landslides WHERE id=?', (case['landslideId'],)).fetchone()
                if (not row or row[0] != case['type'] or registered[case['id']] != case['type']
                        or not case.get('title') or not case.get('citationLocator')):
                    raise ValueError('Missing slide, inconsistent type, or missing case citation: '+case['id'])
                old = db.execute('SELECT landslide_id FROM cases WHERE id=?', (case['id'],)).fetchone()
                if old and old[0] != case['landslideId']:
                    raise ValueError('Cannot reassign existing case identity')
            status = 'ready' if publish else 'draft'
            db.execute('''INSERT INTO documents(id,title,url,citation) VALUES(?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,url=excluded.url,citation=excluded.citation''',
                (source['id'],source['title'],source.get('url'),source.get('note','')))
            for kind in kinds:
                db.execute('''INSERT INTO atlas_types(type,title,description,status) VALUES(?,?,?,?)
                    ON CONFLICT(type) DO UPDATE SET title=excluded.title,description=excluded.description,
                    status=CASE WHEN ? THEN 'ready' ELSE atlas_types.status END''',
                    (kind['type'],kind['name'],kind['quote'],status,publish))
            for case in cases:
                db.execute('''INSERT INTO cases(id,landslide_id,title,description,status) VALUES(?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET title=excluded.title,description=excluded.description,
                    status=CASE WHEN ? THEN 'ready' ELSE cases.status END''',
                    (case['id'],case['landslideId'],case['title'],'\n'.join(case.get('observations',[])),status,publish))
                db.execute('INSERT OR IGNORE INTO case_documents VALUES(?,?,?)',
                    (case['id'],source['id'],case['citationLocator']))
            for kind, entries, key in [('type',kinds,'type'),('case',cases,'id'),('comparison',comparisons,'id')]:
                for entry in entries:
                    db.execute('''INSERT INTO atlas_content VALUES(?,?,?)
                        ON CONFLICT(kind,id) DO UPDATE SET payload_json=excluded.payload_json''',
                        (kind,entry[key],json.dumps(entry,ensure_ascii=False,allow_nan=False)))
            receipt = dict(version=payload['version'],sha256=hashlib.sha256(raw).hexdigest(),
                importedUtc=datetime.now(timezone.utc).isoformat(),publish=publish,
                types=len(kinds),cases=len(cases),comparisons=len(comparisons))
            for key,value in [('atlasContentVersion',payload['version']),('atlasLimits',payload.get('limits',[])),
                              ('atlasImport',receipt)]:
                db.execute('INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                    (key,json.dumps(value,ensure_ascii=False)))
            if db.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('Broken foreign-key references')
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return receipt
