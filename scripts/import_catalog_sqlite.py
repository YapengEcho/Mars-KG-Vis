"""Import a ready contract or stage raw inventory in a new versioned SQLite file."""
import argparse
import json
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
    args=parser.parse_args()
    def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p else None
    result=build_database(args.output,read(args.input),raw=args.raw,source_index=read(args.source_index),
        image_index=read(args.image_index),asset_root=args.asset_root)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
