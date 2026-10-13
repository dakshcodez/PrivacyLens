from __future__ import annotations

import cv2
import numpy as np

from app.schema import RedactionRegion, Style


def redact(frame: np.ndarray, regions: list[RedactionRegion]) -> np.ndarray:
    """Return a redacted copy of frame (BGR). Original is never mutated."""
    out = frame.copy()
    h, w = out.shape[:2]
    for r in regions:
        x0, y0, x1, y1 = r.box.to_pixels(w, h)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        if r.style == Style.BLACK:
            roi[:] = 0
        elif r.style == Style.PIXELATE:
            bh, bw = roi.shape[:2]
            cell = max(8, min(bh, bw) // 3)  # coarse enough that text is unrecoverable
            small = cv2.resize(roi, (max(1, bw // cell), max(1, bh // cell)), interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        else:
            k = max(15, (min(roi.shape[:2]) // 2) | 1)
            k = k if k % 2 else k + 1
            roi[:] = cv2.GaussianBlur(roi, (k, k), 0)
            roi[:] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def failsafe_frame(shape: tuple[int, int], message: str = "PRIVACY LENS: PROTECTION OFFLINE") -> np.ndarray:
    h, w = shape
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = (30, 20, 120)
    scale = max(0.6, w / 1400)
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(f, message, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return f
