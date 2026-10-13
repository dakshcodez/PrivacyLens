"""Internal schema shared by detection, policy and redaction. No raw content is stored here."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    CREDENTIALS = "credentials"
    PRIVATE_CHAT = "private_chat"
    PII = "pii"
    FINANCIAL = "financial"
    SENSITIVE_DOCUMENT = "sensitive_document"


# Categories where a missed mask is costly: masks are sticky and always opaque.
HIGH_RISK = {Category.CREDENTIALS}


@dataclass(frozen=True)
class Box:
    """Normalized (0..1) axis-aligned box: x0,y0 top-left, x1,y1 bottom-right."""
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def validated(x0, y0, x1, y1, min_size: float = 0.002) -> "Box | None":
        try:
            vals = [float(v) for v in (x0, y0, x1, y1)]
        except (TypeError, ValueError):
            return None
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            return None
        x0, y0, x1, y1 = vals
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        # clamp; reject boxes that are entirely outside the frame
        if x1 < 0 or y1 < 0 or x0 > 1 or y0 > 1:
            return None
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (x0, y0, x1, y1))
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            return None
        return Box(x0, y0, x1, y1)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersects(self, o: "Box") -> bool:
        return self.x0 < o.x1 and o.x0 < self.x1 and self.y0 < o.y1 and o.y0 < self.y1

    def iou(self, o: "Box") -> float:
        if not self.intersects(o):
            return 0.0
        iw = min(self.x1, o.x1) - max(self.x0, o.x0)
        ih = min(self.y1, o.y1) - max(self.y0, o.y0)
        inter = iw * ih
        return inter / (self.area + o.area - inter)

    def overlap_of_self(self, o: "Box") -> float:
        """Fraction of *this* box covered by o."""
        if not self.intersects(o):
            return 0.0
        iw = min(self.x1, o.x1) - max(self.x0, o.x0)
        ih = min(self.y1, o.y1) - max(self.y0, o.y0)
        return iw * ih / self.area

    def union(self, o: "Box") -> "Box":
        return Box(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def pad(self, px: float, py: float) -> "Box":
        return Box(max(0.0, self.x0 - px), max(0.0, self.y0 - py), min(1.0, self.x1 + px), min(1.0, self.y1 + py))

    def to_pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        return (int(math.floor(self.x0 * w)), int(math.floor(self.y0 * h)),
                int(math.ceil(self.x1 * w)), int(math.ceil(self.y1 * h)))


@dataclass
class Detection:
    category: Category
    confidence: float
    box: Box
    source: str = "fast"            # fast | gemma
    rule: str = ""                  # rule label, e.g. "jwt" – never the matched text
    mandatory: bool = False         # deterministic high-risk format: masked even if category is OFF in high-security mode
    created: float = field(default_factory=time.time)


class Style(str, Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK = "black"


@dataclass(frozen=True)
class RedactionRegion:
    box: Box
    style: Style
    category: Category
    reason: str = ""
