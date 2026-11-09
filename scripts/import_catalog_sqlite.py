"""Import a ready contract or stage raw inventory in a new versioned SQLite file."""
import argparse
import json
import hashlib
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api.sqlite_store import build_database


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw',action='store_true',help='Stage raw Mars inventory; does not certify coordinates')
    parser.add_argument('--source-index',type=Path)
    parser.add_argument('--image-index',type=Path)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--version', help='Explicit immutable import version')
    parser.add_argument('--coordinate-evidence', help='Source document or confirmed statement; required for a ready real catalog')
    args=parser.parse_args()
    def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p else None
    payload=read(args.input)
    if not args.raw and payload.get('mode') == 'catalog' and not (args.coordinate_evidence or '').strip():
        parser.error('Ready real catalogs require --coordinate-evidence; use --raw for unverified inventory')
    provenance={
        'coordinate_evidence': args.coordinate_evidence,
        'coordinate_action': 'none; importer preserves input values',
        'input_file_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'importer': 'scripts/import_catalog_sqlite.py',
    }
    result=build_database(args.output,payload,raw=args.raw,source_index=read(args.source_index),
        image_index=read(args.image_index),asset_root=args.asset_root,
        version_override=args.version,provenance=provenance)
    with sqlite3.connect(args.output.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        metadata={key:json.loads(value) for key,value in conn.execute('SELECT key,value FROM metadata')}
        result.update(version=metadata['version'], coordinateSystem=metadata['coordinateSystem'],
            provenance=metadata['importProvenance'],
            byType=dict(conn.execute('SELECT type,count(*) FROM landslides GROUP BY type')),
            sources=conn.execute('SELECT count(*) FROM sources').fetchone()[0],
            assets=conn.execute('SELECT count(*) FROM assets').fetchone()[0],
            integrity=conn.execute('PRAGMA integrity_check').fetchone()[0],
            foreign_key_errors=len(conn.execute('PRAGMA foreign_key_check').fetchall()))
    result['database_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
