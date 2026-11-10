"""Small public real-record view. Preserve source coordinates without claiming a verified CRS."""
import json
from pathlib import Path

from api.schemas import Catalog

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_IMAGES = ROOT / 'mobile/public/samples'


def load_public_catalog() -> Catalog:
    source = json.loads((ROOT / 'examples/catalog.real-subset.json').read_text(encoding='utf-8'))
    assets = json.loads((ROOT / 'examples/recognition.samples.json').read_text(encoding='utf-8'))
    images = {row['id']: row['imageUrl'] for row in assets['records']}
    assert source['status'] == 'reference_subset_not_runtime_catalog'
    return Catalog.model_validate({
        'version': 'public-reference-20261110',
        'mode': 'reference',
        'coordinateSystem': {key: source['coordinateStatus'][key] for key in ('body', 'longitude', 'latitude')},
        'records': [dict(row, imageUrl=images.get(row['id'])) for row in source['records']],
    })
