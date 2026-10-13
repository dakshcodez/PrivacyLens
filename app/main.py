"""Entry point: python -m app.main [--backend synthetic --scene whatsapp] [--no-gemma] [--no-vcam]"""
from __future__ import annotations

import argparse
import logging
import os
import signal
from pathlib import Path

import uvicorn
import yaml

from app.pipeline import Pipeline
from app.ui.server import create_app


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("PRIVACY_LENS_CONFIG", "config/default.yaml")
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("OLLAMA_HOST"):
        cfg["gemma"]["host"] = os.environ["OLLAMA_HOST"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--backend", choices=["grim", "mss", "synthetic", "file"])
    ap.add_argument("--scene", help="synthetic scene: terminal|whatsapp|password_manager|document")
    ap.add_argument("--file")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--no-vcam", action="store_true")
    ap.add_argument("--strict-hold", action="store_true")
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    if a.backend:
        cfg["capture"]["backend"] = a.backend
    if a.scene:
        cfg["capture"]["scene"] = a.scene
    if a.file:
        cfg["capture"].update(backend="file", file=a.file)
    if a.no_gemma:
        cfg["gemma"]["enabled"] = False
    if a.no_vcam:
        cfg["output"]["virtual_camera"] = "off"
    if a.strict_hold:
        cfg["pipeline"]["strict_hold"] = True
    if a.port:
        cfg["server"]["port"] = a.port
    pipe = Pipeline(cfg)
    pipe.start()
    srv = cfg["server"]
    print(f"Privacy Lens dashboard: http://{srv['host']}:{srv['port']}")
    try:
        uvicorn.run(create_app(pipe), host=srv["host"], port=srv["port"], log_level="warning")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
