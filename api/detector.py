"""Bounded, offline detection and classification for an oriented CTX crop."""
from dataclasses import dataclass
from io import BytesIO
import math
import os
from pathlib import Path
import re

from PIL import Image

from api.classifier import classification_result, prepare_image, sha256
from api.image_policy import ImageInputError


MAX_DETECTION_SIDE = 2048
MAX_DETECTION_PIXELS = 2048 * 2048
TILE_SIZE = 1024
TILE_STRIDE = 768
DETECTION_SCORE_MIN = 0.20
DETECTION_NMS_IOU = 0.50
MAX_CANDIDATES = 12
MAX_TILE_CANDIDATES = 32
MAX_TILES = 9
ROI_CONTEXT_FRACTION = 0.15
DETECTION_PREPROCESSING_VERSION = 'ctx-tiles1024-stride768-nms50-roi15-v1'


@dataclass(frozen=True)
class DetectionBox:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float


@dataclass(frozen=True)
class PreparedDetection:
    pixels: bytes
    metadata: dict
    preview: bytes


def _check_size(image: Image.Image):
    width, height = image.size
    if not (0 < width <= MAX_DETECTION_SIDE and 0 < height <= MAX_DETECTION_SIDE
            and width * height <= MAX_DETECTION_PIXELS):
        raise ImageInputError('IMAGE_DIMENSIONS_EXCEEDED', 413,
                              'Detection accepts images up to 2048 pixels per side.')


def prepare_detection(image: Image.Image) -> PreparedDetection:
    """Store lossless model input and a metadata-free, same-size JPEG preview."""
    _check_size(image)
    oriented = image.convert('RGB')
    oriented.info.clear()
    pixels, preview = BytesIO(), BytesIO()
    oriented.save(pixels, format='PNG', compress_level=1)
    oriented.save(preview, format='JPEG', quality=90)
    return PreparedDetection(pixels.getvalue(), {
        'width': oriented.width,
        'height': oriented.height,
        'preprocessingVersion': DETECTION_PREPROCESSING_VERSION,
    }, preview.getvalue())


def _tile_origins(length: int) -> list[int]:
    end = max(0, length - TILE_SIZE)
    starts = list(range(0, end + 1, TILE_STRIDE))
    if starts[-1] != end:
        starts.append(end)
    return starts


def _bounded_box(box: DetectionBox, width: int, height: int) -> DetectionBox | None:
    x0, y0, x1, y1, score = map(float, [box.x0, box.y0, box.x1, box.y1, box.score])
    if (not all(math.isfinite(value) for value in [x0, y0, x1, y1, score])
            or not 0 <= score <= 1 or x1 <= x0 or y1 <= y0):
        raise ValueError('Detector returned invalid coordinates or score.')
    # Round outward to keep every source pixel intersecting the prediction.
    left, top = max(0, math.floor(x0)), max(0, math.floor(y0))
    right, bottom = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    if right <= left or bottom <= top:
        return None
    return DetectionBox(left, top, right, bottom, score)


def _iou(a: DetectionBox, b: DetectionBox) -> float:
    overlap = (max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) *
               max(0, min(a.y1, b.y1) - max(a.y0, b.y0)))
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - overlap
    return overlap / union


def _unique_candidates(boxes: list[DetectionBox], width: int, height: int):
    if len(boxes) > MAX_TILES * MAX_TILE_CANDIDATES:
        raise ValueError('Detector returned more than the bounded proposal budget.')
    bounded = []
    for box in boxes:
        candidate = _bounded_box(box, width, height)
        if candidate is not None:
            bounded.append(candidate)
    bounded.sort(key=lambda box: (-box.score, box.x0, box.y0, box.x1, box.y1))

    kept = []
    for candidate in bounded:
        if any(_iou(candidate, earlier) >= DETECTION_NMS_IOU for earlier in kept):
            continue
        if len(kept) == MAX_CANDIDATES:
            return kept, True
        kept.append(candidate)
    return kept, False


class YoloDetector:
    """Load a trusted, SHA-pinned local single-class Ultralytics checkpoint.

    Ultralytics is an optional runtime dependency. Import it only when a detector
    is configured. Operators must use checkpoints from the project's training
    output, since the native .pt loader may deserialize Python objects.
    """

    def __init__(self, weights_path, expected_sha256, version, device='cpu'):
        raw_path = os.fspath(weights_path)
        if '://' in raw_path or "'" in raw_path or raw_path.startswith(('\\\\', '//')):
            raise ValueError('Detector weights must use a local filesystem path.')
        path = Path(raw_path).expanduser().resolve()
        if (str(path).startswith(('\\\\', '//')) or path.suffix != '.pt' or not path.is_file()
                or not isinstance(expected_sha256, str)
                or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256)
                or sha256(path) != expected_sha256.lower()):
            raise ValueError('Detector checkpoint or SHA-256 verification failed.')
        if not isinstance(version, str) or not version.strip() or len(version) > 100:
            raise ValueError('Configure a detector version of 1–100 characters.')
        if device not in ('cpu', 'cuda'):
            raise ValueError('Configure detector device as cpu or cuda.')

        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['ULTRALYTICS_SAFE_LOAD'] = 'true'
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import ONLINE
        from ultralytics.utils.checks import check_yolov5u_filename

        if ONLINE or check_yolov5u_filename(str(path), verbose=False) != str(path):
            raise ValueError('The detector runtime must open the verified checkpoint offline.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('The configured detector CUDA device is unavailable.')
        model = YOLO(str(path))
        if model.task != 'detect' or set(model.names) != {0}:
            raise ValueError('Expected a single-class detection checkpoint.')
        self.model = model
        self.device = device
        self.version = version.strip()
        self.asset_digest = expected_sha256.lower()

    def predict(self, image: Image.Image) -> list[DetectionBox]:
        _check_size(image)
        candidates = []
        for top in _tile_origins(image.height):
            for left in _tile_origins(image.width):
                right, bottom = min(left + TILE_SIZE, image.width), min(top + TILE_SIZE, image.height)
                with image.crop((left, top, right, bottom)) as tile:
                    result, = self.model.predict(
                        source=tile, imgsz=TILE_SIZE, device=self.device,
                        conf=DETECTION_SCORE_MIN, iou=DETECTION_NMS_IOU,
                        max_det=MAX_TILE_CANDIDATES, batch=1, verbose=False, save=False,
                    )
                    boxes = result.boxes.xyxy.cpu().tolist()
                    scores = result.boxes.conf.cpu().tolist()
                    if len(boxes) != len(scores) or len(boxes) > MAX_TILE_CANDIDATES:
                        raise ValueError('Detector returned an invalid number of proposals.')
                    for (x0, y0, x1, y1), score in zip(boxes, scores):
                        bounded = _bounded_box(DetectionBox(x0, y0, x1, y1, score), tile.width, tile.height)
                        if bounded is not None:
                            candidates.append(DetectionBox(bounded.x0 + left, bounded.y0 + top,
                                                           bounded.x1 + left, bounded.y1 + top, bounded.score))
        return candidates


def _candidate_classification(image: Image.Image, box: DetectionBox, classifier) -> dict:
    margin_x = math.ceil((box.x1 - box.x0) * ROI_CONTEXT_FRACTION)
    margin_y = math.ceil((box.y1 - box.y0) * ROI_CONTEXT_FRACTION)
    left, top = max(0, box.x0 - margin_x), max(0, box.y0 - margin_y)
    right, bottom = min(image.width, box.x1 + margin_x), min(image.height, box.y1 + margin_y)
    with image.crop((left, top, right, bottom)) as crop:
        prepared = prepare_image(crop, [box.x0 - left, box.y0 - top, box.x1 - left, box.y1 - top])
    return classification_result(classifier.predict(prepared), classifier)


def detect_and_classify(prepared_png: bytes, detector, classifier) -> dict:
    """Return candidate boxes in source pixels with one classification per box."""
    with Image.open(BytesIO(prepared_png), formats=['PNG']) as image:
        _check_size(image)
        if image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Invalid prepared detector input.')
        image.load()
        boxes, truncated = _unique_candidates(detector.predict(image), image.width, image.height)
        candidates = [
            {
                'candidateId': str(index),
                'box': [box.x0, box.y0, box.x1, box.y1],
                'detectionScore': box.score,
                'classification': _candidate_classification(image, box, classifier),
            }
            for index, box in enumerate(boxes, start=1)
        ]
    return {
        'outcome': 'detections' if candidates else 'no_detections',
        'candidates': candidates,
        'coordinateSpace': 'image_pixels',
        'truncated': truncated,
    }
