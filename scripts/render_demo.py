"""Write before/after images for the synthetic demos to out/ (the 'before' images contain only FAKE data)."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, ".")
from app.capture.sources import SyntheticSource
from app.main import load_config
from app.pipeline import Pipeline

Path("out").mkdir(exist_ok=True)
cfg = load_config()
cfg["output"]["virtual_camera"] = "off"
cfg["gemma"]["enabled"] = "--gemma" in sys.argv
for scene in ("terminal", "password_manager", "whatsapp"):
    src = SyntheticSource(scene)
    p = Pipeline(cfg, src)
    p.start()
    time.sleep(120 if cfg["gemma"]["enabled"] and scene == "whatsapp" else 5)
    cv2.imwrite(f"out/{scene}_before.png", src.read())
    cv2.imwrite(f"out/{scene}_after.png", p.latest_output())
    print(scene, p.status()["protection"], "regions:", p.active_regions)
    p.stop()
