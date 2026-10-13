"""Real-time pipeline: capture -> (fast OCR || Gemma) -> state -> policy -> redaction -> preview / virtual camera.

The output thread never waits on any detector. Detectors run on the latest frame and publish into DetectionState.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import cv2
import numpy as np

from app.capture.sources import Source, make_source
from app.detection.gemma.client import GemmaDetector
from app.detection.gemma.parser import to_detections
from app.detection.ocr.ocr import OcrDetector
from app.detection.refine import refine_credentials
from app.detection.state import DetectionState
from app.monitoring.metrics import Metrics
from app.policy.engine import Policy, PolicyEngine
from app.redaction.redactor import failsafe_frame, redact
from app.virtual_camera.base import make_camera

log = logging.getLogger("privacy_lens")  # log hygiene: only counts/categories/errors, never content


def thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA), (3, 3), 0).astype(np.float32) / 255


def screen_changed(a: np.ndarray | None, b: np.ndarray, frac: float = 0.0004) -> bool:
    return a is None or float((np.abs(a - b) > 0.1).mean()) > frac


class Pipeline:
    def __init__(self, cfg: dict, source: Source | None = None, ocr=None, gemma=None):
        self.cfg = cfg
        pc = cfg["pipeline"]
        self.engine = PolicyEngine(Policy.from_config(cfg["policy"]))
        self.state = DetectionState(pc["stale_scans_high_risk"], pc["stale_scans_normal"], pc["max_age_s"])
        self.metrics = Metrics()
        self.source = source
        self.ocr = ocr if ocr is not None else (OcrDetector(**{k: v for k, v in cfg["ocr"].items() if k != "enabled"}) if cfg["ocr"]["enabled"] else None)
        g = cfg["gemma"]
        self.gemma = gemma if gemma is not None else (GemmaDetector(g["host"], g["model"], g["image_max_side"], g["timeout_s"], g["num_predict"], g["temperature"]) if g["enabled"] else None)
        self.gemma_status = "disabled" if self.gemma is None else "starting"
        self.strict_hold = bool(pc.get("strict_hold", False))
        self.vcam, self.vcam_note = make_camera(cfg["output"]["virtual_camera"], cfg["output"]["device"])
        self.vcam_state = "off"

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._buf: collections.deque = collections.deque(maxlen=max(4, int(cfg["output"]["fps"] * cfg["output"]["delay_ms"] / 1000) + pc["queue_size"] + 2))
        self._latest: tuple[float, np.ndarray] | None = None
        self._capture_error: str | None = None
        self._last_capture = 0.0
        self._fast_last_ok = 0.0
        self._verified_thumb: np.ndarray | None = None
        self._last_safe: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._jpeg_cv = threading.Condition()
        self.protection_state = "starting"
        self.active_regions = 0
        self._out: np.ndarray | None = None
        self._last_picked = 0.0

    # ---------------- lifecycle ----------------
    def start(self):
        if self.source is None:
            self.source = make_source(self.cfg["capture"])
        for fn in (self._capture_loop, self._fast_loop, self._gemma_loop, self._output_loop):
            t = threading.Thread(target=fn, name=fn.__name__, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        with self._jpeg_cv:
            self._jpeg_cv.notify_all()
        deadline = time.time() + 3.0  # shared budget; workers are daemon threads so a stuck HTTP call can't block exit
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.time()))
        if self.source:
            self.source.close()
        self.vcam.close()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def set_source(self, source: Source):
        old, self.source = self.source, source
        self.state.clear()
        self._verified_thumb = None
        self._fast_last_ok = 0.0
        if old:
            old.close()

    # ---------------- workers ----------------
    def _capture_loop(self):
        period = 1 / self.cfg["capture"]["fps"]
        while not self._stop.is_set():
            t0 = time.time()
            try:
                f = self.source.read()
                with self._lock:
                    self._buf.append((t0, f))
                    self._latest = (t0, f)
                    self._last_capture, self._capture_error = t0, None
                self.metrics.capture.tick()
            except Exception as e:
                self._capture_error = type(e).__name__
                time.sleep(0.5)
            self._stop.wait(max(0, period - (time.time() - t0)))

    def _fast_loop(self):
        interval = self.cfg["pipeline"]["fast_interval_ms"] / 1000
        last_ts, last_sig = 0.0, None
        while not self._stop.is_set():
            with self._lock:
                cur = self._latest
            if cur is None or cur[0] == last_ts or self.ocr is None:
                self._stop.wait(0.03)
                if self.ocr is None:
                    self._fast_last_ok = time.time()  # OCR disabled by user: nothing to wait for
                continue
            last_ts, frame = cur
            t0 = time.time()
            try:
                dets = self.ocr.detect(frame)
                self.state.update("fast", dets)
                self._verified_thumb = thumb(frame)
                self._fast_last_ok = time.time()
                self.metrics.ocr_ms.add((self._fast_last_ok - t0) * 1000)
                sig = sorted((d.category.value, d.rule) for d in dets)
                if sig != last_sig:
                    last_sig = sig
                    for c, r in sig:
                        self.metrics.event(c, "detected", "ocr:" + r)
            except Exception as e:
                self.metrics.ocr_fail += 1
                log.warning("fast detector failed: %s", type(e).__name__)
                self._stop.wait(1.0)
            self._stop.wait(max(0, interval - (time.time() - t0)))

    def _gemma_loop(self):
        if self.gemma is None:
            return
        pc = self.cfg["pipeline"]
        last_thumb, last_run, next_avail_check = None, 0.0, 0.0
        while not self._stop.is_set():
            now = time.time()
            if self.gemma_status != "ready" and now >= next_avail_check:
                ok, msg = self.gemma.available()
                self.gemma_status = "ready" if ok else f"unavailable: {msg}"
                next_avail_check = now + 10
            with self._lock:
                cur = self._latest
            if self.gemma_status != "ready" or cur is None or now - last_run < pc["gemma_interval_s"]:
                self._stop.wait(0.2)
                continue
            frame = cur[1]
            th = thumb(frame)
            if not (screen_changed(last_thumb, th, pc["change_threshold"] * 0.05) or now - last_run > pc["gemma_force_interval_s"]):
                self._stop.wait(0.3)
                continue
            last_run, last_thumb = now, th
            self.gemma_status = "inferring"
            try:
                res = self.gemma.classify(frame)
                dets = refine_credentials(to_detections(res), getattr(self.ocr, "last_words", []) if self.ocr else [])
                self.state.update("gemma", dets)  # only successful, validated results touch state
                self.metrics.gemma_ms.add(self.gemma.last_latency_s * 1000)
                self.metrics.gemma_ok += 1
                self.gemma_status = "ready"
                self.metrics.event(res.category.value if res.category and res.sensitive else "none",
                                   "classified" + (" (sensitive)" if res.sensitive else ""), "gemma")
            except Exception as e:  # timeout, HTTP error, malformed output: keep previous masks
                self.metrics.gemma_fail += 1
                self.gemma_status = "ready"
                self.metrics.event("-", "gemma error: " + type(e).__name__, "gemma")
                log.warning("gemma inference failed: %s", type(e).__name__)
                self._stop.wait(2.0)

    # ---------------- output ----------------
    def _pick_frame(self):
        delay = self.cfg["output"]["delay_ms"] / 1000
        with self._lock:
            if not self._buf:
                return None
            cutoff = time.time() - delay
            eligible = [x for x in self._buf if x[0] <= cutoff]
            if not eligible:
                return None
            pick = eligible[-1]
            # frames captured since the last emitted one but skipped (bounded queue: always emit the freshest)
            self.metrics.dropped_frames += sum(1 for x in eligible if self._last_picked < x[0] < pick[0])
            self._last_picked = pick[0]
            return pick

    def _health(self, now: float) -> str:
        if not self.engine.snapshot()["protection_enabled"]:
            return "unprotected"
        if self._capture_error or now - self._last_capture > 2.5 or self._latest is None:
            return "failsafe_capture"
        if self.ocr is not None and now - self._fast_last_ok > max(6.0, 4 * self.cfg["pipeline"]["fast_interval_ms"] / 1000 + 4):
            return "failsafe_detector"  # includes "no verified scan yet"
        if self.gemma is not None and self.gemma_status.startswith("unavailable"):
            return "degraded"
        return "protected"

    def _output_loop(self):
        fps = self.cfg["output"]["fps"]
        period = 1 / fps
        vcam_started = False
        while not self._stop.is_set():
            t0 = time.time()
            item = self._pick_frame()
            health = self._health(t0)
            self.protection_state = health
            try:
                if item is not None or health.startswith("failsafe"):
                    shape = item[1].shape[:2] if item else (720, 1280)
                    if health.startswith("failsafe"):
                        msg = "PRIVACY LENS: STARTING - NOTHING SHOWN YET" if self._fast_last_ok == 0 and health == "failsafe_detector" else "PRIVACY LENS: PROTECTION OFFLINE"
                        out = failsafe_frame(shape, msg)
                        self.active_regions = 0
                    else:
                        ts, frame = item
                        regions = self.engine.apply(self.state.active()) if health != "unprotected" else []
                        self.active_regions = len(regions)
                        if (self.strict_hold and health != "unprotected" and self._last_safe is not None
                                and screen_changed(self._verified_thumb, thumb(frame))):
                            out = self._last_safe  # unverified change on screen: hold last verified frame
                        else:
                            out = redact(frame, regions)
                            self._last_safe = out
                        self.metrics.e2e_ms.add((time.time() - ts) * 1000)
                    self._publish(out, vcam_started)
                    if not vcam_started and self.cfg["output"]["virtual_camera"] != "off":
                        vcam_started = True
                        try:
                            self.vcam.start(out.shape[1], out.shape[0], fps)
                            self.vcam_state = self.vcam.status()
                        except Exception as e:
                            self.vcam_state = f"unavailable ({type(e).__name__}) - preview only"
                    self.metrics.output.tick()
            except Exception as e:
                log.error("output error: %s", type(e).__name__)
            self._stop.wait(max(0, period - (time.time() - t0)))

    def _publish(self, frame: np.ndarray, vcam_started: bool):
        self._out = frame  # redacted/fail-safe frames only
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._jpeg_cv:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1
            self._jpeg_cv.notify_all()
        if vcam_started and self.vcam_state.startswith("streaming"):
            try:
                self.vcam.send(frame)
            except Exception as e:
                self.vcam_state = f"error ({type(e).__name__}) - preview only"

    def latest_output(self):
        return self._out

    def next_jpeg(self, last_seq: int, timeout=1.0):
        with self._jpeg_cv:
            self._jpeg_cv.wait_for(lambda: self._jpeg_seq != last_seq or self._stop.is_set(), timeout)
            return self._jpeg_seq, self._jpeg

    # ---------------- status ----------------
    def status(self) -> dict:
        m = self.metrics.snapshot()
        return {"protection": self.protection_state, "active_regions": self.active_regions, "metrics": m,
                "gemma": {"status": self.gemma_status, "model": self.cfg["gemma"]["model"], "runtime": "ollama (local)"},
                "ocr": "enabled" if self.ocr else "disabled", "virtual_camera": self.vcam_state,
                "capture_error": self._capture_error, "strict_hold": self.strict_hold,
                "policy": self.engine.snapshot(), "source": getattr(self.source, "name", None),
                "scene": getattr(self.source, "scene", None)}
