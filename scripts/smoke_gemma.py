"""Real local-inference smoke test. Requires Ollama running with the model pulled. Prints categories/latency only."""
import sys
import time

sys.path.insert(0, ".")
from app.capture.synthetic import SCENES
from app.detection.gemma.client import GemmaDetector
from app.main import load_config

g = load_config()["gemma"]
det = GemmaDetector(g["host"], g["model"], g["image_max_side"], g["timeout_s"], g["num_predict"], g["temperature"])
ok, msg = det.available()
print(f"runtime available: {ok} ({msg})")
if not ok:
    sys.exit(2)
for name, make in SCENES.items():
    t = time.time()
    r = det.classify(make())
    print(f"{name:17s} -> app={r.app!r} category={r.category.value if r.category else None} sensitive={r.sensitive} "
          f"conf={r.confidence:.2f} regions={[(c.value, None if b is None else tuple(round(v, 2) for v in (b.x0, b.y0, b.x1, b.y1))) for c, b in r.regions]} "
          f"warnings={r.warnings} latency={time.time() - t:.1f}s")
