"""Gemma 4 via a local Ollama server (http://127.0.0.1:11434). Images never leave the machine."""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
import requests

from app.detection.gemma.parser import SCHEMA, GemmaResult, parse_response, to_detections
from app.schema import Detection

PROMPT = """You are a privacy filter for screen sharing. Look at this screenshot and decide whether it shows sensitive content.
Categories:
- credentials: visible passwords, API keys, tokens, password-manager entries, secrets in terminals/editors
- private_chat: private messaging apps (WhatsApp, Telegram, Signal, iMessage, Messenger, Slack/Discord DMs) showing conversations
- pii: personal info (names with addresses, phone numbers, emails, IDs, medical info)
- financial: bank accounts, card numbers, statements, invoices
- sensitive_document: documents marked confidential/internal/private, contracts, HR or legal docs
- none: nothing sensitive (code, web pages, normal documents)
Answer with JSON only. Fields: app (name of the application shown), category (single most important), sensitive (bool),
confidence (0..1), reason (one short sentence, do NOT quote any secret text), regions (list of boxes covering ONLY the sensitive area,
each with category and x0,y0,x1,y1 as fractions of image width/height, origin top-left; for a chat app cover the conversation pane).
If nothing is sensitive: category "none", sensitive false, regions []."""


def encode_frame(frame: np.ndarray, max_side: int) -> str:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


class GemmaDetector:
    name = "gemma"

    def __init__(self, host="http://127.0.0.1:11434", model="gemma4:e2b", image_max_side=640, timeout_s=90,
                 num_predict=700, temperature=0.1, session=None):
        self.host, self.model, self.max_side = host.rstrip("/"), model, image_max_side
        self.timeout, self.num_predict, self.temperature = timeout_s, num_predict, temperature
        self.http = session or requests
        self.last_result: GemmaResult | None = None
        self.last_latency_s: float | None = None

    def available(self) -> tuple[bool, str]:
        try:
            tags = self.http.get(f"{self.host}/api/tags", timeout=3).json()
            names = [m.get("name", "") for m in tags.get("models", [])]
            if not any(n == self.model or n.split(":")[0] == self.model for n in names):
                return False, f"model {self.model} not pulled (run: ollama pull {self.model})"
            return True, "ok"
        except Exception as e:  # connection refused etc.
            return False, f"ollama unreachable: {type(e).__name__}"

    def classify(self, frame: np.ndarray) -> GemmaResult:
        body = {
            "model": self.model, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "30m",
            "messages": [{"role": "user", "content": PROMPT, "images": [encode_frame(frame, self.max_side)]}],
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        t0 = time.time()
        r = self.http.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        self.last_latency_s = time.time() - t0
        self.last_result = parse_response(content)
        return self.last_result

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return to_detections(self.classify(frame))
