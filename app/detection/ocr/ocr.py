"""Fast path: Tesseract word boxes -> text lines -> deterministic detectors -> normalized Detections."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.detection.patterns.secrets import find_matches
from app.schema import Box, Detection


@dataclass
class Word:
    text: str
    box: Box
    line: tuple


def words_from_tsv(data: dict, w: int, h: int, scale: float, min_conf: int) -> list[Word]:
    """Normalize pytesseract image_to_data output to Words with normalized boxes (drops low-confidence/empty)."""
    words = []
    for i in range(len(data["text"])):
        t = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not t or conf < min_conf:
            continue
        x, y, bw, bh = (data[k][i] / scale for k in ("left", "top", "width", "height"))
        box = Box.validated(x / w, y / h, (x + bw) / w, (y + bh) / h, min_size=0.0005)
        if box:
            words.append(Word(t, box, (data["block_num"][i], data["par_num"][i], data["line_num"][i])))
    return words


def group_lines(words: list[Word], aspect: float = 16 / 9) -> list[list[Word]]:
    """Group words into visual lines (same baseline, small horizontal gaps). Independent of tesseract's own line ids."""
    lines: list[list[Word]] = []
    for wd in sorted(words, key=lambda x: (x.box.y0 + x.box.y1) / 2):
        cy, hh = (wd.box.y0 + wd.box.y1) / 2, wd.box.y1 - wd.box.y0
        for ln in lines:
            last = max(ln, key=lambda x: x.box.x1)
            lcy = (last.box.y0 + last.box.y1) / 2
            gap = (wd.box.x0 - last.box.x1) * aspect / hh  # horizontal gap in text-heights
            if abs(cy - lcy) < 0.6 * hh and -1 <= gap < 3:
                ln.append(wd)
                break
        else:
            lines.append([wd])
    return [sorted(ln, key=lambda x: x.box.x0) for ln in lines]


def detect_in_words(words: list[Word], aspect: float = 16 / 9) -> list[Detection]:
    """Run text detectors per line and map matched character spans back to word boxes."""
    out: list[Detection] = []
    for lw in group_lines(words, aspect):
        text, spans, pos = "", [], 0
        for wd in lw:
            if text:
                text += " "
                pos += 1
            spans.append((pos, pos + len(wd.text), wd))
            text += wd.text
            pos += len(wd.text)
        for m in find_matches(text):
            hit = [wd for s, e, wd in spans if s < m.end and m.start < e]
            if not hit:
                continue
            box = hit[0].box
            for wd in hit[1:]:
                box = box.union(wd.box)
            out.append(Detection(m.category, m.confidence, box, "fast", m.rule, m.mandatory))
    return out


class OcrDetector:
    name = "ocr"

    def __init__(self, lang="eng", min_word_conf=30, upscale_below_width=1600):
        self.lang, self.min_conf, self.upscale_w = lang, min_word_conf, upscale_below_width
        self.last_words: list[Word] = []  # in-memory only, never logged

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = 1.0
        img = frame
        if w < self.upscale_w:
            scale = min(2.0, self.upscale_w / w)
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 110:  # light-on-dark (terminals): tesseract prefers dark-on-light
            gray = cv2.bitwise_not(gray)
        data = pytesseract.image_to_data(gray, lang=self.lang, config="--psm 11", output_type=pytesseract.Output.DICT)
        words = words_from_tsv(data, w, h, scale, self.min_conf)
        self.last_words = words
        return detect_in_words(words, w / h)
