from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.capture.sources import CaptureError, SyntheticSource, list_windows, make_source
from app.capture.synthetic import SCENES
from app.pipeline import Pipeline

from pathlib import Path

STATIC = Path(__file__).parent / "static"


class CaptureReq(BaseModel):
    backend: str | None = None
    scope: str | None = None
    window: str | None = None
    scene: str | None = None


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="Privacy Lens")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return pipeline.status()

    @app.post("/api/policy")
    def policy(d: dict):
        pipeline.engine.update(d)
        if "strict_hold" in d:
            pipeline.strict_hold = bool(d["strict_hold"])
        return pipeline.engine.snapshot()

    @app.get("/api/windows")
    def windows():
        return {"windows": list_windows(), "scenes": list(SCENES)}

    @app.post("/api/capture")
    def capture(r: CaptureReq):
        c = pipeline.cfg["capture"]
        for k in ("backend", "scope", "window"):
            if getattr(r, k) is not None:
                c[k] = getattr(r, k)
        if r.scene and c["backend"] == "synthetic" and isinstance(pipeline.source, SyntheticSource):
            pipeline.source.set_scene(r.scene)
            return {"ok": True}
        c["scene"] = r.scene or c.get("scene")
        try:
            pipeline.set_source(make_source(c))
        except (CaptureError, Exception) as e:
            raise HTTPException(400, f"capture failed: {type(e).__name__}")
        return {"ok": True}

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        async def gen():
            seq = 0
            loop = asyncio.get_running_loop()
            while not pipeline.stopping and not await request.is_disconnected():
                seq, jpg = await loop.run_in_executor(None, pipeline.next_jpeg, seq, 1.0)
                if jpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    return app
