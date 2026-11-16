"""One inference worker per database; durable jobs survive HTTP disconnects."""
import json
import logging
import os
from pathlib import Path
import threading

from api.classifier import PreparedImage, classification_result

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
    def __init__(self, store, model):
        self.store, self.model = store, model
        self.lock = WorkerLock(store.path)
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None

    @property
    def available(self):
        return self.model is not None and self.thread is not None and self.thread.is_alive() and not self.stop.is_set()

    def start(self):
        self.lock.acquire()
        try:
            self.store.recover_interrupted()
            if self.model is not None:
                self.thread = threading.Thread(target=self.run, name='mars-classifier', daemon=True)
                self.thread.start()
        except BaseException:
            self.lock.release()
            raise

    def run(self):
        try:
            while not self.stop.is_set():
                row = self.store.claim()
                if row is None:
                    self.wake.wait(.25)
                    self.wake.clear()
                    continue
                try:
                    prepared = PreparedImage(bytes(row['pixels']), json.loads(row['weights_json']),
                                             json.loads(row['input_json']), bytes(row['preview']))
                    result = classification_result(self.model.predict(prepared), self.model)
                except Exception:
                    # Do not send exception text, filesystem paths or submitted image data to clients.
                    logger.warning('Classification failed for task %s', row['id'])
                    self.store.finish(row['id'], error='INFERENCE_FAILED')
                else:
                    self.store.finish(row['id'], result=result)
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
