"""Synthetic screens with FAKE data only: terminal credentials, WhatsApp-like chat, password manager, plain doc."""
from __future__ import annotations

import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720

# Obviously fake, but format-valid, so deterministic detectors fire.
FAKE_AWS = "AKIAIOSFODNN7EXAMPLE"
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlLXVzZXIifQ.Zm9vYmFyYmF6cXV4MTIz"
FAKE_GH = "ghp_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5"
FAKE_EMAIL = "jane.doe@example.com"


def _font(size: int, bold=False, mono=False):
    pats = (["/usr/share/fonts/**/DejaVuSansMono*.ttf", "/usr/share/fonts/**/NotoSansMono-Regular.ttf", "/usr/share/fonts/**/*Mono*.ttf"]
            if mono else ["/usr/share/fonts/**/LiberationSans-%s.ttf" % ("Bold" if bold else "Regular"),
                          "/usr/share/fonts/**/DejaVuSans.ttf"])
    for p in pats:
        for f in sorted(glob.glob(p, recursive=True)):
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _to_bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def terminal_scene() -> np.ndarray:
    img = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    f = _font(22, mono=True)
    lines = ["$ cat .env", "DATABASE_HOST=db.internal", f"AWS_ACCESS_KEY_ID={FAKE_AWS}",
             f"export GITHUB_TOKEN={FAKE_GH}", f"Authorization: Bearer {FAKE_JWT}", "$ ls -la", "drwxr-xr-x  src  docs  tests"]
    for i, ln in enumerate(lines):
        d.text((30, 40 + i * 36), ln, font=f, fill=(220, 225, 230))
    return _to_bgr(img)


def whatsapp_scene() -> np.ndarray:
    img = Image.new("RGB", (W, H), (240, 242, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(0, 128, 105))
    d.text((24, 14), "WhatsApp Web", font=_font(26, True), fill="white")
    d.rectangle([0, 56, 380, H], fill="white")
    names = [("Mom", "Don't tell dad about the surprise"), ("Alex", "ok see you at 7"), ("Sam (work)", "salary doc attached"), ("Priya", "call me when free")]
    for i, (n, m) in enumerate(names):
        y = 70 + i * 82
        d.ellipse([16, y + 8, 64, y + 56], fill=(180, 190, 195))
        d.text((82, y + 8), n, font=_font(20, True), fill=(20, 20, 20))
        d.text((82, y + 36), m, font=_font(16), fill=(110, 110, 110))
    d.rectangle([380, 56, W, H], fill=(229, 221, 213))
    d.rectangle([380, 56, W, 116], fill=(240, 242, 245))
    d.text((410, 72), "Mom   online", font=_font(20, True), fill=(20, 20, 20))
    msgs = [(False, "Did you get the test results?"), (True, "Yes, doctor says it's fine"), (False, "Thank god. Love you"),
            (True, "Don't tell anyone yet please"), (False, "Of course. Dinner Sunday?")]
    y = 140
    for mine, t in msgs:
        tw = d.textlength(t, font=_font(20)) + 32
        x0 = W - 40 - tw if mine else 410
        d.rounded_rectangle([x0, y, x0 + tw, y + 48], 12, fill=(217, 253, 211) if mine else "white")
        d.text((x0 + 16, y + 12), t, font=_font(20), fill=(20, 20, 20))
        y += 70
    d.rectangle([380, H - 64, W, H], fill=(240, 242, 245))
    d.rounded_rectangle([400, H - 52, W - 40, H - 12], 20, fill="white")
    d.text((420, H - 42), "Type a message", font=_font(18), fill=(140, 140, 140))
    return _to_bgr(img)


def password_manager_scene() -> np.ndarray:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=(26, 115, 232))
    d.text((24, 18), "Password Manager", font=_font(28, True), fill="white")
    d.text((120, 130), "example-bank.test", font=_font(30, True), fill=(30, 30, 30))
    d.text((120, 200), "Username", font=_font(20), fill=(100, 100, 100))
    d.text((120, 230), FAKE_EMAIL, font=_font(26), fill=(30, 30, 30))
    d.text((120, 300), "Password", font=_font(20), fill=(100, 100, 100))
    d.text((120, 330), "password: Tr0ub4dor&3xplo!t", font=_font(26), fill=(30, 30, 30))
    d.text((120, 420), "Show   Copy   Edit", font=_font(22), fill=(26, 115, 232))
    return _to_bgr(img)


def document_scene() -> np.ndarray:
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((80, 60), "Quarterly planning notes", font=_font(36, True), fill=(20, 20, 20))
    for i, t in enumerate(["Agenda for the offsite: roadmap, hiring, tooling.", "Please review the draft before Friday.",
                           "Nothing sensitive is on this page."]):
        d.text((80, 140 + i * 44), t, font=_font(24), fill=(40, 40, 40))
    return _to_bgr(img)


SCENES = {"terminal": terminal_scene, "whatsapp": whatsapp_scene, "password_manager": password_manager_scene, "document": document_scene}
