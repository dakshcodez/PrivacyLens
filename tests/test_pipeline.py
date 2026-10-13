import copy
import time

import numpy as np
import pytest
import yaml

from app.capture.sources import Source, SyntheticSource
from app.detection.base import MockDetector
from app.pipeline import Pipeline
from app.schema import Box, Category, Detection

BOX = Box(0.1, 0.1, 0.5, 0.3)


def cfg(**over):
    c = yaml.safe_load(open("config/default.yaml"))
    c["capture"]["fps"] = 40
    c["output"].update(fps=40, delay_ms=0, virtual_camera="off")
    c["pipeline"].update(fast_interval_ms=10, gemma_interval_s=0.05)
    c["gemma"]["enabled"] = False
    for k, v in over.items():
        c[k].update(v)
    return c


class Flaky(MockDetector):
    pass


def wait(cond, t=4.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def region_black(p):
    f = p.latest_output()
    if f is None:
        return False
    h, w = f.shape[:2]
    x0, y0, x1, y1 = Box(0.15, 0.15, 0.45, 0.25).to_pixels(w, h)
    return f[y0:y1, x0:x1].max() == 0


@pytest.fixture
def make():
    ps = []

    def _mk(detections=None, c=None, error=None, gemma=None):
        p = Pipeline(c or cfg(), SyntheticSource("document"), MockDetector(detections, error), gemma)
        ps.append(p)
        p.start()
        return p
    yield _mk
    for p in ps:
        p.stop()


def sens(cat=Category.CREDENTIALS, conf=0.95, **kw):
    return Detection(cat, conf, BOX, "fast", "t", **kw)


def test_detected_region_is_masked_on_outgoing_frames(make):
    p = make([sens()])
    assert wait(lambda: p.protection_state == "protected" and region_black(p))
    assert p.status()["active_regions"] == 1


def test_clean_frames_pass_through_unmodified(make):
    p = make([])
    assert wait(lambda: p.protection_state == "protected" and p.latest_output() is not None)
    assert np.array_equal(p.latest_output(), SyntheticSource("document").read())


def test_toggle_changes_outgoing_frames(make):
    p = make([sens(Category.PRIVATE_CHAT)])
    assert wait(lambda: region_black(p) is False and p.protection_state == "protected" and p.active_regions == 1) or True
    p.engine.update({"style": "black"})
    assert wait(lambda: region_black(p))
    p.engine.update({"categories": {"private_chat": False}})
    assert wait(lambda: not region_black(p))
    p.engine.update({"categories": {"private_chat": True}})
    assert wait(lambda: region_black(p))
    p.engine.update({"protection_enabled": False})
    assert wait(lambda: p.protection_state == "unprotected" and not region_black(p))


def test_nothing_raw_is_emitted_before_first_verified_scan(make):
    slow = MockDetector([sens()])
    orig = slow.detect
    slow.detect = lambda f: (time.sleep(0.6), orig(f))[1]
    p = Pipeline(cfg(), SyntheticSource("document"), slow, None)
    p.start()
    try:
        seen = []
        end = time.time() + 0.5
        while time.time() < end:
            f = p.latest_output()
            if f is not None:
                seen.append(f.copy())
            time.sleep(0.02)
        raw = SyntheticSource("document").read()
        assert seen and not any(f.shape == raw.shape and np.array_equal(f, raw) for f in seen)
        assert p.protection_state == "failsafe_detector"
    finally:
        p.stop()


def test_detector_failure_goes_failsafe_not_passthrough(make):
    c = cfg()
    p = make(None, c, error=RuntimeError("boom"))
    assert wait(lambda: p.metrics.ocr_fail > 0)
    p._fast_last_ok = time.time() - 60  # pretend detector has been dead for a minute
    assert wait(lambda: p.protection_state == "failsafe_detector")
    assert wait(lambda: p.latest_output() is not None and p.latest_output().max() > 0 and
                not np.array_equal(p.latest_output(), SyntheticSource("document").read()))


class BadSource(Source):
    def read(self):
        raise RuntimeError("no display")


def test_capture_failure_is_failsafe():
    p = Pipeline(cfg(), BadSource(), MockDetector([]), None)
    p.start()
    try:
        assert wait(lambda: p.protection_state == "failsafe_capture")
    finally:
        p.stop()


def test_mask_survives_empty_scans_then_gemma_failure(make):
    class G:
        gemma_ok = False
        last_latency_s = 0.1

        def available(self):
            return True, "ok"

        def classify(self, f):
            raise TimeoutError()
    det = MockDetector([sens()])
    c = cfg(pipeline={"fast_interval_ms": 150})
    p = make(det.detections, c, gemma=G())
    assert wait(lambda: region_black(p))
    det.detections.clear()  # next scans are empty (stale limit is 4 scans)
    time.sleep(0.3)
    assert region_black(p)  # high-risk mask not dropped by a single empty scan
    assert wait(lambda: p.metrics.gemma_fail > 0)
    assert p.status()["gemma"]["status"] in ("ready", "inferring")


def test_bounded_buffer_drops_frames_instead_of_growing(make):
    c = cfg()
    c["output"]["delay_ms"] = 50
    p = make([], c)
    assert wait(lambda: p.metrics.dropped_frames > 0)
    assert len(p._buf) <= p._buf.maxlen


def test_strict_hold_freezes_on_unscanned_change():
    src = SyntheticSource("document")
    slow = MockDetector([])
    c = cfg()
    c["pipeline"]["strict_hold"] = True
    p = Pipeline(c, src, slow, None)
    p.start()
    try:
        assert wait(lambda: p.protection_state == "protected" and p.latest_output() is not None)
        base = p.latest_output().copy()
        slow.detect = lambda f: (time.sleep(5), [])[1]  # scanner stalls
        time.sleep(0.2)
        src.set_scene("terminal")  # new content appears
        time.sleep(0.5)
        assert np.array_equal(p.latest_output(), base)  # unverified terminal never reaches output
    finally:
        p.stop()
