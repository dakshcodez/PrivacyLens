import numpy as np
import pytest

from app.capture.synthetic import SCENES
from app.detection.ocr.ocr import OcrDetector, Word, detect_in_words, group_lines, words_from_tsv
from app.schema import Box, Category


def w(text, x0, y0, x1, y1):
    return Word(text, Box(x0, y0, x1, y1), (0, 0, 0))


def test_tsv_normalization_drops_noise_and_scales():
    data = {"text": ["", "ok", "bad", "x" * 14], "conf": ["-1", "90", "10", "5"],
            "left": [0, 200, 0, 400], "top": [0, 100, 0, 100], "width": [10, 200, 10, 400], "height": [10, 40, 10, 40],
            "block_num": [0] * 4, "par_num": [0] * 4, "line_num": [0] * 4}
    ws = words_from_tsv(data, 1000, 500, 2.0, 30)
    assert [x.text for x in ws] == ["ok", "x" * 14]  # long low-conf token is kept
    assert ws[0].box == Box(0.1, 0.1, 0.2, 0.14)


def test_words_grouped_into_lines_and_boxes_cover_match_only():
    ws = [w("key", 0.1, 0.1, 0.15, 0.13), w("AKIAIOSFODNN7EXAMPLE", 0.16, 0.1, 0.4, 0.13), w("other", 0.1, 0.5, 0.2, 0.53)]
    assert len(group_lines(ws)) == 2
    ds = detect_in_words(ws)
    assert len(ds) == 1 and ds[0].category == Category.CREDENTIALS
    assert ds[0].box.y1 < 0.2 and ds[0].box.x0 == 0.16


@pytest.mark.slow
def test_ocr_finds_planted_fakes_on_synthetic_terminal():
    ds = OcrDetector().detect(SCENES["terminal"]())
    assert {"aws_access_key", "github_token"} <= {d.rule for d in ds}
    assert all(0 <= d.box.x0 < d.box.x1 <= 1 and 0 <= d.box.y0 < d.box.y1 <= 1 for d in ds)


@pytest.mark.slow
def test_ocr_no_detections_on_plain_document():
    assert OcrDetector().detect(SCENES["document"]()) == []
