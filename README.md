# Privacy Lens

A local AI privacy firewall for screen sharing. It captures your screen, finds sensitive regions (secrets, private chats,
PII, finance, documents), redacts them, and publishes only the **redacted** video as a preview and a virtual camera.
All analysis is local: OCR + deterministic detectors for speed, **Gemma 4 (via Ollama, on-device)** for semantic understanding.

> Core principle: redaction happens *before* the video reaches Meet/Zoom/Teams. Nothing is ever patched after sending.

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
sudo pacman -S tesseract tesseract-data-eng grim   # Arch/Wayland; apt: tesseract-ocr grim
# Gemma 4 runtime (local)
ollama serve &            # or: systemctl enable --now ollama
ollama pull gemma4:e2b    # ~4.6 GB, vision-capable, CPU-friendly
# run
.venv/bin/python -m app.main                      # real screen (grim on Wayland, --backend mss elsewhere)
.venv/bin/python -m app.main --backend synthetic --scene whatsapp   # repeatable fake-data demo
```
Open http://127.0.0.1:8765. Options: `--no-gemma`, `--no-vcam`, `--strict-hold`, `--file clip.mp4`.
Settings live in `config/default.yaml`; categories, style, sensitivity and strict hold are also live in the dashboard.

### Virtual camera (Linux)
```bash
sudo pacman -S v4l2loopback-dkms linux-headers && ./scripts/setup_virtual_camera.sh   # creates /dev/video10 "Privacy Lens"
```
Then pick **Privacy Lens** as the camera in Meet. Set `output.device` if you want a specific node. If the device is missing the
app keeps running with the preview only and says so in the dashboard.

**This is not Meet's "Present now".** Meet's native screen share uses the browser's `getDisplayMedia()` which nothing outside the
browser can intercept. You share the redacted feed *as your camera* instead (your real webcam is replaced while you do).
Windows/macOS need a virtual-camera bridge (e.g. OBS Virtual Camera, which pyvirtualcam supports); only Linux was targeted here.

## How it decides
| Layer | What | Notes |
|---|---|---|
| Fast path | Tesseract word boxes → line grouping → regex for known key formats, JWT, cards (Luhn), emails, phones; `password=`-style assignments; entropy-based unknown tokens | ~0.3–0.6 s per frame on this CPU |
| Gemma 4 | Screenshot (≤640 px side) → JSON schema `{app, category, sensitive, confidence, reason, regions[]}` → validated | async, on screen change / cadence |
| Policy | per-category toggle, sensitivity (0.85/0.6/0.35), uncertain credentials still masked, high-security mandatory masks | `app/policy/engine.py`, pure logic |
| Redaction | black / pixelate / blur (credentials always black); large masks labelled by category | `app/redaction/redactor.py` |

See [docs/architecture.md](docs/architecture.md). Model choice: `gemma4:e2b` (Gemma 4 edge size with image input) via Ollama, chosen because
it runs on CPU-only machines; set `gemma.model` (e.g. `gemma4:e4b`) to trade latency for accuracy. Any runtime can be swapped by
implementing `detect(frame) -> list[Detection]`.

## Measured on the dev machine (14 threads, no GPU, 15 GB RAM; Ollama CPU)
Real values from `scripts/benchmark.py --scene whatsapp --gemma --seconds 60` and `scripts/smoke_gemma.py`:

| metric | value |
|---|---|
| capture / output FPS | 15.0 / 15.0 (configured cap 15) |
| OCR pass (while Gemma is running) | ~600 ms (≈440 ms when idle) |
| Gemma 4 E2B latency, 640 px | 8–22 s per inference (13.9 s / 12.4 s avg in the run above) |
| end-to-end local latency (capture→emit) | ~270 ms (includes the 250 ms hold-back buffer) |
A GPU would shorten Gemma latency; none was available to measure.

What Gemma got right on the synthetic scenes (real model): terminal & password manager → `credentials`, WhatsApp-like → `private_chat`,
plain document → not sensitive. **Localization is coarse:** for the chat it returns the whole window, for the terminal it boxed the wrong
lines. So Gemma supplies *what* and OCR supplies *where* for secrets; chats are masked over the whole captured area (sidebar previews are private too).
Use `capture.scope: window` (Hyprland) to limit the mask to just the chat window.

## Privacy guarantees (and their limits)
* Frames, OCR text and model output are processed in memory only. Nothing is written to disk; logs and the detection log hold
  category, time, action and rule name — never content. Screenshots are only sent to `gemma.host` (default `127.0.0.1`).
  If you point that at a remote machine, your screen leaves your computer; don't do that with real screens.
* Detection is asynchronous, so **newly appearing content can be visible for up to one scan (≈0.5 s + `delay_ms`) before a mask lands.**
  `output.delay_ms` hides most of that; **Strict hold** closes it for the OCR path by freezing on the last verified frame
  while the screen has changed and is unscanned. Gemma-only categories (chats) can lag by a full inference (~10 s) the first
  time they appear. Nothing here is a zero-exposure guarantee.
* Fail-safe: before the first scan, if OCR dies for several seconds, or if capture fails, the output is a solid "offline" frame — not raw video.
  Protection OFF is the only way raw video is sent and the dashboard says "UNPROTECTED".
* Masks persist: a credential mask needs 4 consecutive successful scans without re-detection to expire; failed/timeout Gemma
  calls never remove masks.

## Limitations
* Passwords shown as images, in video, or briefly (<1 scan) are not caught; masked dots (`••••`) have no text, so only Gemma's coarse region may flag them.
* OCR misses tiny/low-contrast text and mangles very long tokens (handled by keeping low-confidence long words, still imperfect).
* Entropy-based "unknown" secrets use a 0.7 confidence and may hit random IDs; word-level boxes mask the whole word.
* Gemma E2B on CPU is slow and its boxes are unreliable; classification was good on synthetic scenes only — **not evaluated on real WhatsApp Web / password managers.**
* Window capture is Hyprland-only (`hyprctl`); other compositors use whole screen. Wayland capture via `grim` is a screenshot loop (~15 fps here).
* Virtual camera is **unverified in this environment** until `v4l2loopback` is loaded (see Troubleshooting).

## Tests
```bash
.venv/bin/python -m pytest                       # 56 tests; no Gemma needed (mock detectors); OCR tests use real tesseract
.venv/bin/python -m pytest -m smoke -o addopts=''  # real local Gemma (ollama must be running)
```

## Troubleshooting
* **Dashboard says `FAIL-SAFE: detector not ready`** — OCR hasn't completed (is `tesseract` installed?) or crashed.
* **`Gemma: unavailable: ollama unreachable`** — start `ollama serve`; `model … not pulled` → `ollama pull gemma4:e2b`. The app keeps working (OCR only, "degraded").
* **Ollama 500 `llama-server binary not found`** — incomplete tarball install; reinstall the package.
* **No virtual camera** — `modinfo v4l2loopback`; then `./scripts/setup_virtual_camera.sh` (needs sudo; `exclusive_caps=1` is required for Chrome). Check `v4l2-ctl --list-devices`.
* **Black screen via grim** — ensure you're on wlroots/Hyprland; otherwise use `--backend mss`.

See [docs/demo.md](docs/demo.md) for the demo script and status checklist.
