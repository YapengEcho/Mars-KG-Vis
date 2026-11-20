"""One inference worker per database; durable jobs survive HTTP disconnects."""
import json
import logging
import math
import os
from pathlib import Path
import threading

from api.classifier import PreparedImage, classification_result
from api.detector import detect_and_classify
from api.image_policy import decode_image, is_ctx_grayscale

logger = logging.getLogger(__name__)


class WorkerLock:
    def __init__(self, database):
        self.path = Path(str(database) + '.worker.lock')
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        try:
            if stream.seek(0, 2) == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            stream.close()
            raise RuntimeError('Recognition database already has a worker; use one API process.') from error
        self.stream = stream

    def release(self):
        if self.stream is not None:
            self.stream.close()  # OS releases the advisory lock even after process termination.
            self.stream = None


class RecognitionWorker:
    def __init__(self, store, model, detector=None, *, binary_screen=None):
        self.store, self.model = store, model
        self.detector = detector
        self.binary_screen = binary_screen
        self.lock = WorkerLock(store.path)
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None

    @property
    def _model_worker_ready(self):
        return (self.model is not None and
                self.thread is not None and self.thread.is_alive() and not self.stop.is_set())

    @property
    def available(self):
        return self.binary_screen is not None and self._model_worker_ready

    @property
    def detection_available(self):
        return self.detector is not None and self._model_worker_ready

    def start(self):
        self.lock.acquire()
        try:
            self.store.recover_interrupted()
            if self.model is not None and (self.binary_screen is not None or self.detector is not None):
                self.thread = threading.Thread(target=self.run, name='mars-classifier', daemon=True)
                self.thread.start()
        except BaseException:
            self.lock.release()
            raise

    def screen(self, image, metadata):
        if not is_ctx_grayscale(image):
            return dict(decision='unsupported_image', score=None, threshold=None, modelVersion=None)
        roi = metadata['roi'] if metadata['roiSource'] == 'provided_box' else None
        score = float(self.binary_screen.predict(image, roi=roi))
        threshold = float(self.binary_screen.threshold)
        if not math.isfinite(threshold):
            raise ValueError('Screening threshold must be finite.')
        if not math.isfinite(score):
            return dict(decision='inconclusive', score=None, threshold=threshold,
                        modelVersion=self.binary_screen.version)
        decision = 'possible_landslide' if score >= threshold else 'no_landslide'
        return dict(decision=decision, score=score, threshold=threshold, modelVersion=self.binary_screen.version)

    def run(self):
        try:
            while not self.stop.is_set():
                row = self.store.claim()
                if row is None:
                    self.wake.wait(.25)
                    self.wake.clear()
                    continue
                try:
                    screening = None
                    if row['kind'] == 'detect':
                        result = detect_and_classify(bytes(row['pixels']), self.detector, self.model)
                    else:
                        metadata = json.loads(row['input_json'])
                        screening = self.screen(decode_image(bytes(row['screen_input'])), metadata)
                        result = None
                        if screening['decision'] == 'possible_landslide':
                            prepared = PreparedImage(bytes(row['pixels']), json.loads(row['weights_json']),
                                                     metadata, bytes(row['preview']))
                            result = classification_result(self.model.predict(prepared), self.model)
                except Exception:
                    # Do not send exception text, filesystem paths or submitted image data to clients.
                    logger.warning('Inference failed for task %s', row['id'])
                    self.store.finish(row['id'], error='INFERENCE_FAILED')
                else:
                    self.store.finish(row['id'], result=result, screening=screening)
        except Exception:
            logger.exception('Recognition worker stopped; restart the API to recover pending jobs.')
        finally:
            self.lock.release()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)
            # A live inference keeps its lock until it finishes. Never start a competing worker.
        else:
            self.lock.release()
