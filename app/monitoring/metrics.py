"""Rolling metrics; real measurements only. Never stores frame content."""
from __future__ import annotations

import collections
import threading
import time


class Rate:
    def __init__(self, window=3.0):
        self.t = collections.deque()
        self.window = window

    def tick(self):
        now = time.time()
        self.t.append(now)
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()

    def fps(self) -> float:
        now = time.time()
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()
        return len(self.t) / self.window if len(self.t) > 1 else 0.0


class Series:
    def __init__(self, n=50):
        self.v = collections.deque(maxlen=n)
        self.last = None

    def add(self, x: float):
        self.v.append(x)
        self.last = x

    def avg(self):
        return sum(self.v) / len(self.v) if self.v else None


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture, self.output = Rate(), Rate()
        self.ocr_ms, self.gemma_ms, self.e2e_ms = Series(), Series(), Series()
        self.dropped_frames = 0
        self.gemma_ok = self.gemma_fail = self.ocr_fail = 0
        self.log: collections.deque = collections.deque(maxlen=60)

    def event(self, category: str, action: str, source: str = ""):
        with self.lock:
            self.log.appendleft({"t": time.strftime("%H:%M:%S"), "category": category, "action": action, "source": source})

    def snapshot(self) -> dict:
        r = lambda x: None if x is None else round(x, 1)
        with self.lock:
            return {"capture_fps": round(self.capture.fps(), 1), "output_fps": round(self.output.fps(), 1),
                    "ocr_ms": r(self.ocr_ms.avg()), "gemma_ms": r(self.gemma_ms.last), "gemma_ms_avg": r(self.gemma_ms.avg()),
                    "e2e_ms": r(self.e2e_ms.avg()), "dropped_frames": self.dropped_frames,
                    "gemma_ok": self.gemma_ok, "gemma_fail": self.gemma_fail, "ocr_fail": self.ocr_fail,
                    "log": list(self.log)[:25]}
