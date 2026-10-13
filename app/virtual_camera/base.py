from __future__ import annotations

import numpy as np


class VirtualCamera:
    """Output backend interface. The redaction pipeline only ever calls these methods."""
    name = "null"

    def start(self, width: int, height: int, fps: int) -> None: ...
    def send(self, frame_bgr: np.ndarray) -> None: ...
    def close(self) -> None: ...

    def status(self) -> str:
        return "off"


class NullCamera(VirtualCamera):
    pass


class PyVirtualCam(VirtualCamera):
    name = "pyvirtualcam"

    def __init__(self, device: str | None = None):
        self.device, self.cam = device, None

    def start(self, width, height, fps):
        import pyvirtualcam
        self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, device=self.device,
                                       fmt=pyvirtualcam.PixelFormat.BGR)
        self.w, self.h = width, height

    def send(self, frame_bgr):
        if self.cam is None:
            return
        if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
            import cv2
            frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
        self.cam.send(frame_bgr)

    def close(self):
        if self.cam:
            self.cam.close()
            self.cam = None

    def status(self):
        return f"streaming to {self.cam.device}" if self.cam else "off"


def make_camera(mode: str, device: str | None):
    """auto: try pyvirtualcam, silently fall back to preview-only. Returns (camera, human-readable note)."""
    if mode == "off":
        return NullCamera(), "virtual camera disabled"
    cam = PyVirtualCam(device)
    return cam, "pyvirtualcam"
