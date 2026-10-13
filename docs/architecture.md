# Architecture

```
capture thread ──► ring buffer (bounded, delay_ms) ─────────────────────────────┐
      │                                                                          ▼
      ├─► fast worker: Tesseract words → line grouping → patterns/entropy ─► DetectionState ─► PolicyEngine ─► redact() ─► MJPEG preview
      │                                                  (thread-safe, persistent)   (live toggles)      │            └► virtual camera
      └─► Gemma worker (async, only on screen change / cadence): Ollama → JSON → validate → OCR-refine ─┘
```

* **Output never waits for a detector.** The output thread runs at `output.fps`, takes the freshest frame that is at least
  `delay_ms` old, applies the *current* policy to the *current* detection state and emits. Toggling a category therefore
  changes the next outgoing frame.
* **DetectionState** keeps tracks per source (`fast`, `gemma`). A track is dropped only after N *successful* scans in which
  nothing covered it (4 for credentials, 2 otherwise) or after `max_age_s`. A failed/timed-out Gemma call never touches state.
* **PolicyEngine** is pure logic: category toggle, sensitivity threshold, conservative masking of uncertain credentials
  (threshold × 0.6), high-security mandatory masks, overlap merging. Credentials are always opaque black.
* **Gemma** receives a downscaled screenshot (`gemma.image_max_side`), answers with a JSON-schema-constrained object, and the reply is
  validated (`detection/gemma/parser.py`): malformed JSON, bad confidence, missing / out-of-range / degenerate coordinates are handled.
  Coordinates in 0–1, 0–100 or 0–1000 are accepted. Sensitive-but-unlocalized results fall back to a conservative content-area mask.
  For credentials, Gemma's coarse region is tightened with OCR word boxes (`detection/refine.py`).
* **Runtime is swappable:** anything with `detect(frame) -> list[Detection]` can replace `GemmaDetector`.

## Protection states
| state | meaning | output |
|---|---|---|
| `protected` | capture, OCR and (if enabled) Gemma healthy | redacted video |
| `degraded` | Gemma unavailable | redacted by OCR/patterns only |
| `unprotected` | user turned protection off | raw video, clearly flagged |
| `failsafe_detector` | no verified scan yet, or OCR has failed for several seconds | solid "offline" frame |
| `failsafe_capture` | capture lost | solid "offline" frame |

`pipeline.strict_hold` (dashboard: *Strict hold*) additionally re-emits the last verified frame while the screen has visibly changed
and no scan of the new content has completed yet. This is the only mode that closes the "new secret appears before detection" window
for the deterministic path; it costs smoothness while content is changing.

## Why a virtual camera is not Meet's "Share screen"
Google Meet's *Present now* uses the browser's `getDisplayMedia()` API, which captures the screen inside the browser. Nothing on
this machine can sit in front of that. Privacy Lens instead publishes the **already redacted** frames as a camera device
("Privacy Lens"); you choose it as your camera (or present it as a window/tab that shows the preview) in Meet/Zoom/Teams.
The receiving app only ever sees redacted pixels. The side effect is that your actual camera is replaced by the screen feed while
you use it.
