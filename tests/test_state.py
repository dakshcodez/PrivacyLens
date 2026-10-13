from app.detection.state import DetectionState
from app.schema import Box, Category, Detection

A = Box(0.1, 0.1, 0.3, 0.2)


def d(cat=Category.CREDENTIALS, box=A, source="fast"):
    return Detection(cat, 0.9, box, source)


def test_detection_persists_until_enough_empty_scans_high_risk():
    s = DetectionState(stale_high=4, stale_normal=2)
    s.update("fast", [d()])
    for i in range(3):
        s.update("fast", [])  # empty results must not instantly unmask
        assert len(s.active()) == 1, i
    s.update("fast", [])
    assert s.active() == []


def test_normal_risk_expires_sooner():
    s = DetectionState(stale_high=4, stale_normal=2)
    s.update("fast", [d(Category.PII)])
    s.update("fast", [])
    assert len(s.active()) == 1
    s.update("fast", [])
    assert s.active() == []


def test_redetection_refreshes_and_follows_movement():
    s = DetectionState(stale_high=3)
    s.update("fast", [d()])
    s.update("fast", [])
    moved = Box(0.1, 0.5, 0.3, 0.6)
    s.update("fast", [d(box=moved)])
    assert moved in [x.box for x in s.active()]  # new location masked immediately
    assert A in [x.box for x in s.active()]      # old one lingers (conservative) ...
    s.update("fast", [d(box=moved)])
    assert [x.box for x in s.active()] == [moved]  # ... then ages out


def test_failed_scan_leaves_state_untouched_and_sources_are_independent():
    s = DetectionState()
    s.update("gemma", [d(Category.PRIVATE_CHAT, source="gemma")])
    s.update("fast", [])  # fast scans don't age gemma tracks
    for _ in range(5):
        s.update("fast", [])
    assert len(s.active()) == 1


def test_max_age_hard_expiry():
    s = DetectionState(max_age_s=10)
    s.update("fast", [d()], now=100.0)
    assert len(s.active(now=105.0)) == 1
    assert s.active(now=111.0) == []
