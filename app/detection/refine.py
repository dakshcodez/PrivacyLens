"""Combine Gemma's coarse region with OCR word boxes for tight credential masks."""
from __future__ import annotations

import re

from app.detection.ocr.ocr import Word
from app.schema import Box, Category, Detection


def _valuey(t: str) -> bool:
    t = t.strip()
    return len(t) >= 6 and not t.endswith(":") and bool(re.search(r"\d|[^\w\s]", t)) and not t.startswith(("http", "www"))


def refine_credentials(gemma_dets: list[Detection], words: list[Word]) -> list[Detection]:
    out: list[Detection] = []
    for d in gemma_dets:
        if d.category != Category.CREDENTIALS or d.rule != "model_localized":
            out.append(d)
            continue
        inside = [w for w in words if w.box.overlap_of_self(d.box) > 0.6 and _valuey(w.text)]
        if inside:
            out += [Detection(Category.CREDENTIALS, d.confidence, w.box, "gemma", "gemma_ocr_refined") for w in inside]
        else:
            out.append(d)
    return out
