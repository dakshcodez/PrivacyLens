"""Real OCR + policy + redaction on synthetic fake-credential screens (no Gemma)."""
import time

import numpy as np
import pytest
import yaml

from app.capture.sources import SyntheticSource
from app.pipeline import Pipeline

pytestmark = pytest.mark.slow


def run(scene, **policy):
    c = yaml.safe_load(open("config/default.yaml"))
    c["output"].update(delay_ms=0, virtual_camera="off")
    c["gemma"]["enabled"] = False
    c["policy"].update(policy)
    src = SyntheticSource(scene)
    p = Pipeline(c, src)
    p.start()
    try:
        end = time.time() + 8
        while time.time() < end and not (p.protection_state == "protected" and p.active_regions):
            time.sleep(0.1)
        time.sleep(0.5)
        return src.read(), p.latest_output(), p
    finally:
        p.stop()


def test_terminal_credentials_masked_rest_intact():
    raw, out, p = run("terminal")
    assert p.active_regions >= 1
    changed = (raw != out).any(axis=2)
    assert changed[110:140, 30:500].any()          # the AWS key row
    assert not changed[40:100, 30:360].any()        # "$ cat .env" / DATABASE_HOST untouched
    assert not changed[400:, :].any()


def test_pii_toggle_changes_output_on_password_manager():
    raw, out_on, _ = run("password_manager")
    raw, out_off, _ = run("password_manager", categories={"credentials": False, "pii": False}, high_security=False)
    assert not np.array_equal(out_on, raw)
    assert np.array_equal(out_off, raw)
