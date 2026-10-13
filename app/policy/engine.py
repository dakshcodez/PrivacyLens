"""Policy engine: pure logic, independent of UI, model and capture."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.schema import HIGH_RISK, Box, Category, Detection, RedactionRegion, Style

SENSITIVITY_THRESHOLD = {"low": 0.85, "medium": 0.6, "high": 0.35}
# High-risk categories are masked when merely *uncertain* (conservative masking).
UNCERTAIN_FACTOR = 0.6
PAD = 0.004  # normalized padding around masks


@dataclass
class Policy:
    protection_enabled: bool = True
    high_security: bool = True
    style: Style = Style.BLACK
    sensitivity: str = "medium"
    categories: dict = field(default_factory=lambda: {c.value: True for c in Category})

    @staticmethod
    def from_config(cfg: dict) -> "Policy":
        p = Policy()
        p.update(cfg)
        return p

    def update(self, d: dict) -> None:
        if "protection_enabled" in d:
            self.protection_enabled = bool(d["protection_enabled"])
        if "high_security" in d:
            self.high_security = bool(d["high_security"])
        if "style" in d and d["style"] in {s.value for s in Style}:
            self.style = Style(d["style"])
        if d.get("sensitivity") in SENSITIVITY_THRESHOLD:
            self.sensitivity = d["sensitivity"]
        for k, v in (d.get("categories") or {}).items():
            if k in self.categories:
                self.categories[k] = bool(v)

    def to_dict(self) -> dict:
        return {"protection_enabled": self.protection_enabled, "high_security": self.high_security,
                "style": self.style.value, "sensitivity": self.sensitivity, "categories": dict(self.categories)}


class PolicyEngine:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            return self.policy.to_dict()

    def update(self, d: dict) -> None:
        with self.lock:
            self.policy.update(d)

    def decide(self, det: Detection) -> Style | None:
        """Return the redaction style for one detection, or None if it must not be redacted."""
        with self.lock:
            p = self.policy
            if Box.validated(det.box.x0, det.box.y0, det.box.x1, det.box.y1) is None:
                return None
            if not p.protection_enabled:
                return None
            thr = SENSITIVITY_THRESHOLD[p.sensitivity]
            enabled = p.categories.get(det.category.value, False)
            if enabled:
                floor = thr * UNCERTAIN_FACTOR if det.category in HIGH_RISK else thr
                if det.confidence >= floor:
                    return Style.BLACK if det.category in HIGH_RISK else p.style
            # Category OFF: only explicit mandatory high-risk formats survive, and only in high-security mode.
            if p.high_security and det.mandatory and det.category in HIGH_RISK and det.confidence >= 0.9:
                return Style.BLACK
            return None

    def apply(self, detections: list[Detection]) -> list[RedactionRegion]:
        regions = []
        for d in detections:
            style = self.decide(d)
            if style is not None:
                regions.append(RedactionRegion(d.box.pad(PAD, PAD), style, d.category, d.rule))
        return merge_regions(regions)


_RANK = {Style.BLUR: 0, Style.PIXELATE: 1, Style.BLACK: 2}


def merge_regions(regions: list[RedactionRegion]) -> list[RedactionRegion]:
    """Union overlapping regions of the same style; a region fully inside a stronger one is dropped."""
    regs = list(regions)
    changed = True
    while changed:
        changed = False
        out: list[RedactionRegion] = []
        for r in regs:
            for i, o in enumerate(out):
                if o.style == r.style and o.box.intersects(r.box):
                    out[i] = RedactionRegion(o.box.union(r.box), o.style, o.category, o.reason)
                    changed = True
                    break
            else:
                out.append(r)
        regs = out
    final = [r for r in regs
             if not any(o is not r and _RANK[o.style] > _RANK[r.style] and r.box.overlap_of_self(o.box) > 0.95 for o in regs)]
    return sorted(final, key=lambda r: _RANK[r.style])  # strongest drawn last
