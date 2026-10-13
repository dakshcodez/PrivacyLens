"""Thread-safe persistent detection state with conservative expiry."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.schema import HIGH_RISK, Detection

MATCH_IOU = 0.3


@dataclass
class Track:
    det: Detection
    last_seen: float
    misses: int = 0


class DetectionState:
    def __init__(self, stale_high=4, stale_normal=2, max_age_s=120.0):
        self._tracks: dict[str, list[Track]] = {"fast": [], "gemma": []}
        self._lock = threading.Lock()
        self.stale_high, self.stale_normal, self.max_age_s = stale_high, stale_normal, max_age_s

    def update(self, source: str, detections: list[Detection], now: float | None = None) -> None:
        """Merge the result of a *successful* scan from `source`. Failed scans must not call this."""
        now = now or time.time()
        with self._lock:
            old = self._tracks.setdefault(source, [])
            new_tracks: list[Track] = []
            matched: set[int] = set()
            for d in detections:
                best, bi = 0.0, -1
                for i, t in enumerate(old):
                    if i in matched or t.det.category != d.category:
                        continue
                    iou = t.det.box.iou(d.box)
                    if iou > best:
                        best, bi = iou, i
                if bi >= 0 and best >= MATCH_IOU:
                    matched.add(bi)
                new_tracks.append(Track(d, now, 0))
            for i, t in enumerate(old):
                if i in matched:
                    continue
                # not matched at its location – is the box still covered by a new detection of the same category?
                covered = any(d.category == t.det.category and t.det.box.overlap_of_self(d.box) > 0.5 for d in detections)
                if covered:
                    continue
                t.misses += 1
                limit = self.stale_high if t.det.category in HIGH_RISK else self.stale_normal
                if t.misses < limit and now - t.last_seen < self.max_age_s:
                    new_tracks.append(t)
            self._tracks[source] = new_tracks

    def active(self, now: float | None = None) -> list[Detection]:
        now = now or time.time()
        with self._lock:
            return [t.det for ts in self._tracks.values() for t in ts if now - t.last_seen < self.max_age_s]

    def clear(self) -> None:
        with self._lock:
            for k in self._tracks:
                self._tracks[k] = []
