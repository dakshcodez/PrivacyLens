import json

import pytest
import requests

from app.detection.gemma.client import GemmaDetector
from app.detection.gemma.parser import GemmaParseError, parse_response, to_detections
from app.schema import Box, Category


def resp(**kw):
    base = {"app": "WhatsApp Web", "category": "private_chat", "sensitive": True, "confidence": 0.93, "reason": "chat",
            "regions": [{"category": "private_chat", "x0": 0.3, "y0": 0.1, "x1": 1.0, "y1": 1.0}]}
    base.update(kw)
    return json.dumps(base)


def test_valid_response_to_detection():
    ds = to_detections(parse_response(resp()))
    assert len(ds) == 1 and ds[0].category == Category.PRIVATE_CHAT and ds[0].source == "gemma"
    assert ds[0].box == Box(0.3, 0.1, 1.0, 1.0)


def test_fenced_and_chatty_output():
    assert parse_response("```json\n" + resp() + "\n```").sensitive
    assert parse_response("Sure! Here you go: " + resp() + " Hope that helps").category == Category.PRIVATE_CHAT


@pytest.mark.parametrize("bad", ["", "not json", "[1,2]", "{broken", '"str"'])
def test_malformed_raises(bad):
    with pytest.raises(GemmaParseError):
        parse_response(bad)


def test_not_sensitive_yields_nothing():
    assert to_detections(parse_response(resp(sensitive=False, category="none", regions=[]))) == []


def test_missing_and_invalid_coordinates_fall_back_conservatively():
    r = parse_response(resp(regions=[{"category": "private_chat"}]))
    assert "region missing coordinates" in r.warnings
    d = to_detections(r)[0]
    assert d.rule == "model_fallback_fullscreen" and d.box.area > 0.9 and d.confidence < 0.93
    r = parse_response(resp(regions=[{"category": "private_chat", "x0": 5000, "y0": 9000, "x1": 6000, "y1": 9500}]))
    assert to_detections(r)[0].rule == "model_fallback_fullscreen"


def test_sensitive_without_regions_is_still_masked():
    ds = to_detections(parse_response(resp(regions=[])))
    assert len(ds) == 1 and ds[0].rule == "model_fallback_fullscreen"


def test_out_of_range_values_are_clamped_or_rejected():
    r = parse_response(resp(confidence=7, regions=[{"category": "private_chat", "x0": -0.2, "y0": 0.1, "x1": 1.4, "y1": 0.9}]))
    assert r.confidence == 1.0
    assert r.regions[0][1] == Box(0.0, 0.1, 1.0, 0.9)
    assert parse_response(resp(confidence="high")).confidence == 0.0


def test_percent_and_permille_coordinates():
    p = parse_response(resp(regions=[{"category": "chat", "x0": 30, "y0": 10, "x1": 100, "y1": 100}]))
    assert p.regions[0][1] == Box(0.3, 0.1, 1.0, 1.0)
    m = parse_response(resp(regions=[{"category": "chat", "x0": 300, "y0": 100, "x1": 1000, "y1": 1000}]))
    assert m.regions[0][1] == Box(0.3, 0.1, 1.0, 1.0)


def test_unknown_category_dropped():
    r = parse_response(resp(category="weather", regions=[{"category": "weather", "x0": 0, "y0": 0, "x1": .5, "y1": .5}]))
    assert r.regions == [] and to_detections(r) == []


class FakeHttp:
    def __init__(self, content=None, exc=None, status=200):
        self.content, self.exc, self.status = content, exc, status

    def post(self, *a, **k):
        if self.exc:
            raise self.exc
        outer = self

        class R:
            def raise_for_status(s):
                if outer.status != 200:
                    raise requests.HTTPError("boom")

            def json(s):
                return {"message": {"content": outer.content}}
        return R()

    def get(self, *a, **k):
        if self.exc:
            raise self.exc
        return type("R", (), {"json": lambda s: {"models": [{"name": "gemma4:e2b"}]}})()


def frame():
    import numpy as np
    return np.zeros((100, 200, 3), np.uint8)


def test_client_roundtrip_and_failures():
    g = GemmaDetector(session=FakeHttp(resp()))
    assert g.detect(frame())[0].category == Category.PRIVATE_CHAT and g.last_latency_s is not None
    assert g.available()[0]
    for bad in (FakeHttp(exc=requests.Timeout()), FakeHttp(status=500), FakeHttp(content="garbage")):
        with pytest.raises(Exception):
            GemmaDetector(session=bad).detect(frame())
    ok, msg = GemmaDetector(session=FakeHttp(exc=requests.ConnectionError())).available()
    assert not ok and "unreachable" in msg
