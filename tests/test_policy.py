import pytest

from app.policy.engine import Policy, PolicyEngine, merge_regions
from app.schema import Box, Category, Detection, RedactionRegion, Style

B = Box(0.1, 0.1, 0.3, 0.2)


def det(cat, conf=0.95, box=B, mandatory=False):
    return Detection(cat, conf, box, "fast", "t", mandatory)


def eng(**kw):
    p = Policy()
    p.update(kw)
    return PolicyEngine(p)


def test_category_toggle():
    e = eng(style="blur")
    assert e.decide(det(Category.PRIVATE_CHAT)) == Style.BLUR
    e.update({"categories": {"private_chat": False}})
    assert e.decide(det(Category.PRIVATE_CHAT)) is None
    assert e.decide(det(Category.PII)) == Style.BLUR  # others unaffected


def test_credentials_always_black():
    assert eng(style="blur").decide(det(Category.CREDENTIALS)) == Style.BLACK


@pytest.mark.parametrize("sens,conf,expected", [("low", 0.7, None), ("medium", 0.7, Style.PIXELATE), ("high", 0.4, Style.PIXELATE),
                                                ("high", 0.2, None)])
def test_sensitivity_thresholds(sens, conf, expected):
    assert eng(style="pixelate", sensitivity=sens).decide(det(Category.PII, conf)) == expected


def test_uncertain_high_risk_is_masked_conservatively():
    e = eng(sensitivity="medium")  # threshold .6, high-risk floor .36
    assert e.decide(det(Category.CREDENTIALS, 0.4)) == Style.BLACK
    assert e.decide(det(Category.PII, 0.4)) is None


def test_high_security_keeps_mandatory_masks_when_category_off():
    e = eng(categories={"credentials": False})
    assert e.decide(det(Category.CREDENTIALS, 0.97, mandatory=True)) == Style.BLACK
    assert e.decide(det(Category.CREDENTIALS, 0.97, mandatory=False)) is None
    e.update({"high_security": False})
    assert e.decide(det(Category.CREDENTIALS, 0.97, mandatory=True)) is None


def test_master_switch_and_invalid_coordinates():
    e = eng(protection_enabled=False)
    assert e.decide(det(Category.CREDENTIALS, mandatory=True)) is None
    bad = Detection(Category.PII, 0.9, Box(0.5, 0.5, 0.5, 0.5))
    assert eng().decide(bad) is None


def test_coordinate_validation():
    assert Box.validated(0.3, 0.3, 0.1, 0.1) == Box(0.1, 0.1, 0.3, 0.3)  # swapped
    assert Box.validated(-0.2, 0.1, 0.5, 0.4) == Box(0, 0.1, 0.5, 0.4)  # clamped
    assert Box.validated(1.2, 0, 1.5, 1) is None  # outside
    assert Box.validated(float("nan"), 0, 1, 1) is None
    assert Box.validated("a", 0, 1, 1) is None
    assert Box.validated(0.1, 0.1, 0.1001, 0.5) is None  # degenerate


def test_overlapping_regions_merge_and_strongest_wins():
    a = RedactionRegion(Box(0.1, 0.1, 0.3, 0.3), Style.BLACK, Category.CREDENTIALS)
    b = RedactionRegion(Box(0.2, 0.2, 0.4, 0.4), Style.BLACK, Category.CREDENTIALS)
    m = merge_regions([a, b])
    assert len(m) == 1 and m[0].box == Box(0.1, 0.1, 0.4, 0.4)
    inner = RedactionRegion(Box(0.12, 0.12, 0.2, 0.2), Style.BLUR, Category.PII)
    assert merge_regions([a, inner]) == [a]
    far = RedactionRegion(Box(0.6, 0.6, 0.7, 0.7), Style.BLUR, Category.PII)
    assert len(merge_regions([a, far])) == 2


def test_apply_end_to_end():
    e = eng()
    regs = e.apply([det(Category.PII), det(Category.PII, box=Box(0.25, 0.15, 0.5, 0.3)), det(Category.PII, 0.1, Box(.7, .7, .8, .8))])
    assert len(regs) == 1
