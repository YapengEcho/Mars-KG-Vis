"""Import reviewed atlas.json text into an existing catalog database."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api.content_store import import_atlas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--publish',action='store_true',help='Explicitly mark reviewed text ready; does not approve coordinates')
    args=parser.parse_args()
    print(json.dumps(import_atlas(args.database,args.input,publish=args.publish),ensure_ascii=False))


if __name__=='__main__':main()
