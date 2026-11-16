"""Private task database, separate from the read-only geological catalog."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class QueueFull(Exception):
    pass


class TaskStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS recognition_sessions (
                    token_hash TEXT PRIMARY KEY, created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recognition_jobs (
                    id TEXT PRIMARY KEY, owner_hash TEXT NOT NULL REFERENCES recognition_sessions(token_hash),
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    created_at TEXT NOT NULL, finished_at TEXT, model_version TEXT NOT NULL,
                    preprocessing_version TEXT NOT NULL, input_json TEXT NOT NULL,
                    pixels BLOB, weights_json TEXT, preview BLOB NOT NULL,
                    result_json TEXT, error_code TEXT);
                CREATE INDEX IF NOT EXISTS recognition_owner ON recognition_jobs(owner_hash,created_at);
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def issue_session(self):
        token = secrets.token_urlsafe(32)
        expiry = (datetime.now(timezone.utc)+timedelta(days=90)).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO recognition_sessions VALUES (?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(), now(), expiry))
        return dict(token=token, tokenType='Bearer', expiresAt=expiry)

    def authenticate(self, token):
        if not isinstance(token, str) or not 40 <= len(token) <= 128:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            found = db.execute('SELECT 1 FROM recognition_sessions WHERE token_hash=? AND expires_at>?',
                               (digest,now())).fetchone()
        return digest if found else None

    def enqueue(self, owner, prepared, model_version, max_pending=8):
        task_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT count(*) FROM recognition_jobs WHERE status IN ('queued','running')").fetchone()[0] >= max_pending:
                raise QueueFull()
            db.execute('''INSERT INTO recognition_jobs
                (id,owner_hash,status,created_at,model_version,preprocessing_version,input_json,pixels,weights_json,preview)
                VALUES (?,?,'queued',?,?,?,?,?,?,?)''',
                (task_id,owner,now(),model_version,prepared.metadata['preprocessingVersion'],
                 json.dumps(prepared.metadata),prepared.pixels,json.dumps(prepared.weights),prepared.preview))
        return task_id

    def recover_interrupted(self):
        with self.connect() as db:
            db.execute("""UPDATE recognition_jobs SET status='failed',finished_at=?,error_code='PROCESS_RESTARTED',
                pixels=NULL,weights_json=NULL WHERE status IN ('queued','running')""", (now(),))

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM recognition_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE recognition_jobs SET status='running' WHERE id=?", (row['id'],))
            return row

    def finish(self, task_id, result=None, error=None):
        with self.connect() as db:
            db.execute('''UPDATE recognition_jobs SET status=?,finished_at=?,result_json=?,error_code=?,
                pixels=NULL,weights_json=NULL WHERE id=? AND status='running' ''',
                ('failed' if error else 'completed',now(),json.dumps(result,allow_nan=False) if result else None,error,task_id))

    @staticmethod
    def public(row):
        return dict(taskId=row['id'],kind='classify',status=row['status'],createdAt=row['created_at'],
            finishedAt=row['finished_at'],modelVersion=row['model_version'],
            preprocessingVersion=row['preprocessing_version'],input=json.loads(row['input_json']),
            result=json.loads(row['result_json']) if row['result_json'] else None,
            errorCode=row['error_code'],imageUrl=f'/api/recognition/tasks/{row["id"]}/image')

    def get(self, owner, task_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM recognition_jobs WHERE id=? AND owner_hash=?', (task_id,owner)).fetchone()
            return self.public(row) if row else None

    def list(self, owner, page, page_size):
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM recognition_jobs WHERE owner_hash=?',(owner,)).fetchone()[0]
            rows = db.execute('SELECT * FROM recognition_jobs WHERE owner_hash=? ORDER BY rowid DESC LIMIT ? OFFSET ?',
                              (owner,page_size,(page-1)*page_size)).fetchall()
            return dict(items=[self.public(r) for r in rows],total=total,page=page,pageSize=page_size)

    def image(self, owner, task_id):
        with self.connect() as db:
            row = db.execute('SELECT preview FROM recognition_jobs WHERE id=? AND owner_hash=?',(task_id,owner)).fetchone()
            return bytes(row[0]) if row else None

    def delete(self, owner, task_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM recognition_jobs WHERE id=? AND owner_hash=?',(task_id,owner)).fetchone()
            if row is None:
                return 'missing'
            if row[0] in ('queued','running'):
                return 'busy'
            db.execute('DELETE FROM recognition_jobs WHERE id=? AND owner_hash=?',(task_id,owner))
            return 'deleted'
