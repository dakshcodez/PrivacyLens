from __future__ import annotations

from typing import Protocol

import numpy as np

from app.schema import Detection


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a BGR frame. Raise on failure (callers keep prior state)."""
        ...


class MockDetector:
    """Test double: returns canned detections, or raises."""
    name = "mock"

    def __init__(self, detections=None, error: Exception | None = None):
        self.detections, self.error, self.calls = detections or [], error, 0

    def detect(self, frame):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.detections)
