"""In-memory invocation trace recorder.
The skeleton keeps traces in memory; persistence to SQLite follows in week 2.
"""
import time
import uuid
from collections import deque
class TraceRecorder:
    def __init__(self, maxlen: int = 200):
        self._traces: deque = deque(maxlen=maxlen)
    def new_id(self) -> str:
        return "trc-" + uuid.uuid4().hex[:12]
    def record(self, trace_id: str, data: dict) -> None:
        entry = dict(data)
        entry["ts"] = time.time()
        self._traces.append((trace_id, entry))
    def list(self) -> list:
        return list(self._traces)
