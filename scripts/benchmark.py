"""Run the real pipeline on a synthetic scene and print *measured* numbers."""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")
from app.main import load_config
from app.pipeline import Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="terminal")
ap.add_argument("--seconds", type=int, default=30)
ap.add_argument("--gemma", action="store_true")
a = ap.parse_args()
cfg = load_config()
cfg["capture"].update(backend="synthetic", scene=a.scene)
cfg["output"]["virtual_camera"] = "off"
cfg["gemma"]["enabled"] = a.gemma
p = Pipeline(cfg)
p.start()
time.sleep(a.seconds)
s = p.status()
p.stop()
m = s["metrics"]
m.pop("log")
print(json.dumps({"scene": a.scene, "seconds": a.seconds, "protection": s["protection"], "regions": s["active_regions"], "gemma": s["gemma"]["status"], **m}, indent=1))
