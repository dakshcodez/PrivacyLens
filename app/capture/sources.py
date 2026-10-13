from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from app.capture.synthetic import SCENES


class CaptureError(RuntimeError):
    pass


def _limit_width(f: np.ndarray, max_w: int) -> np.ndarray:
    if max_w and f.shape[1] > max_w:
        s = max_w / f.shape[1]
        f = cv2.resize(f, (max_w, int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return f


class Source:
    name = "base"

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class SyntheticSource(Source):
    """Fake-data scene. `scene` can be switched live (used by the demo and the dashboard)."""
    name = "synthetic"

    def __init__(self, scene="terminal"):
        self.scene = scene
        self._cache: dict[str, np.ndarray] = {}

    def set_scene(self, scene: str):
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene}")
        self.scene = scene

    def read(self):
        if self.scene not in self._cache:
            self._cache[self.scene] = SCENES[self.scene]()
        return self._cache[self.scene].copy()


class FileSource(Source):
    name = "file"

    def __init__(self, path: str):
        img = cv2.imread(path)
        self.img, self.cap = img, None
        if img is None:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise CaptureError(f"cannot open {path}")

    def read(self):
        if self.img is not None:
            return self.img.copy()
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
            if not ok:
                raise CaptureError("video read failed")
        return f

    def close(self):
        if self.cap:
            self.cap.release()


class GrimSource(Source):
    """Wayland (wlroots/Hyprland) capture via `grim`. Window scope uses `hyprctl clients`."""
    name = "grim"

    def __init__(self, geometry: str | None = None, max_width=0):
        if not shutil.which("grim"):
            raise CaptureError("grim not installed")
        self.geometry, self.max_width = geometry, max_width

    def read(self):
        cmd = ["grim", "-t", "ppm"] + (["-g", self.geometry] if self.geometry else []) + ["-"]
        p = subprocess.run(cmd, capture_output=True, timeout=5)
        if p.returncode != 0 or not p.stdout:
            raise CaptureError("grim failed")
        f = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        if f is None:
            raise CaptureError("grim returned undecodable data")
        return _limit_width(f, self.max_width)


class MssSource(Source):
    name = "mss"

    def __init__(self, monitor=1, max_width=0):
        import mss
        self.sct, self.monitor, self.max_width = mss.mss(), monitor, max_width

    def read(self):
        shot = self.sct.grab(self.sct.monitors[self.monitor])
        return _limit_width(np.asarray(shot)[:, :, :3].copy(), self.max_width)


def list_windows() -> list[dict]:
    """Visible windows on Hyprland: [{id, title, geometry}]. Titles are shown only in the local dashboard."""
    if not shutil.which("hyprctl"):
        return []
    try:
        out = json.loads(subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, timeout=3).stdout)
    except Exception:
        return []
    ws = []
    for c in out:
        if c.get("mapped") and not c.get("hidden") and c["size"][0] > 50:
            (x, y), (w, h) = c["at"], c["size"]
            ws.append({"id": c["address"], "title": f'{c.get("class", "")}: {c.get("title", "")}'[:70],
                       "geometry": f"{x},{y} {w}x{h}"})
    return ws


def make_source(cfg: dict) -> Source:
    b = cfg.get("backend", "grim")
    if b == "synthetic":
        return SyntheticSource(cfg.get("scene") or "terminal")
    if b == "file":
        return FileSource(cfg["file"])
    if b == "mss":
        return MssSource(max_width=cfg.get("max_width", 0))
    geometry = None
    if cfg.get("scope") == "window" and cfg.get("window"):
        geometry = next((w["geometry"] for w in list_windows() if w["id"] == cfg["window"]), None)
        if geometry is None:
            raise CaptureError("selected window not found")
    return GrimSource(geometry, cfg.get("max_width", 0))
