import numpy as np

from app.redaction.redactor import failsafe_frame, redact
from app.schema import Box, Category, RedactionRegion, Style


def noisy():
    return np.random.default_rng(0).integers(0, 255, (200, 400, 3), dtype=np.uint8)


def reg(style):
    return RedactionRegion(Box(0.25, 0.25, 0.75, 0.75), style, Category.PII)


def test_black_mask_is_opaque_and_outside_untouched():
    f = noisy()
    o = redact(f, [reg(Style.BLACK)])
    assert o[50:150, 100:300].max() == 0
    assert np.array_equal(o[:40], f[:40]) and np.array_equal(o[:, :90], f[:, :90])
    assert o is not f and not np.array_equal(f, o)  # original not mutated


def test_blur_and_pixelate_destroy_detail_but_not_black():
    f = noisy()
    for s in (Style.BLUR, Style.PIXELATE):
        o = redact(f, [reg(s)])
        roi_in, roi_out = f[50:150, 100:300].astype(float), o[50:150, 100:300].astype(float)
        assert roi_out.std() < roi_in.std() * 0.5
        assert roi_out.max() > 0


def test_full_frame_and_edge_regions_do_not_crash():
    f = noisy()
    assert redact(f, [RedactionRegion(Box(0, 0, 1, 1), Style.BLACK, Category.CREDENTIALS)]).max() == 0
    redact(f, [RedactionRegion(Box(0.99, 0.99, 1, 1), Style.BLUR, Category.PII)])
    assert np.array_equal(redact(f, []), f)


def test_failsafe_frame_shape_and_no_source_content():
    ff = failsafe_frame((120, 300))
    assert ff.shape == (120, 300, 3)


def test_black_mask_output_independent_of_source_pixels_even_with_label():
    a, b = noisy(), np.random.default_rng(1).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    r = [RedactionRegion(Box(0, 0, 1, 1), Style.BLACK, Category.PRIVATE_CHAT)]
    oa, ob = redact(a, r), redact(b, r)
    assert np.array_equal(oa, ob)  # nothing of the source survives
    big = np.zeros((400, 800, 3), np.uint8)
    assert redact(big, r).max() > 0  # label drawn on large masks
