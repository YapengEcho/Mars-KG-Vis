"""Create a private manifest for an explicitly chosen, frozen classifier bundle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.classifier import CLASSES, PREPROCESSING_VERSION, load_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Local DINOv2-base directory')
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--normalizer', type=Path, required=True)
    parser.add_argument('--version', required=True, help='Frozen model version, not an accuracy claim')
    parser.add_argument('--output', type=Path, required=True, help='Private JSON manifest; keep outside tracked sources')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new versioned manifest path.')
    assets = {'baseConfig': args.base/'config.json', 'baseWeights': args.base/'model.safetensors',
              'adapter': args.adapter, 'normalizer': args.normalizer}
    spec = dict(schemaVersion=1, architecture='dinov2-base-qv-last2-r4', classes=list(CLASSES),
                modelVersion=args.version, preprocessingVersion=PREPROCESSING_VERSION,
                files={name: dict(path=str(path.resolve()), sha256=sha256(path)) for name,path in assets.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(spec, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    load_manifest(args.output)
    print('Created and verified private classifier manifest.')


if __name__ == '__main__':
    main()
