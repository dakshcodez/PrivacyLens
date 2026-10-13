# Demo script (~3 minutes, fake data only)

1. `ollama serve` (in another shell), then `python -m app.main --backend synthetic --scene terminal`, open http://127.0.0.1:8765.
2. **Credentials:** a fake terminal with an AWS key, GitHub token and bearer JWT appears. Within ~0.5 s those words turn into black
   bars; `DATABASE_HOST=…`, `ls` output stay readable. Point at the log: category / time / action only.
3. Switch *Demo scene* → **password_manager**: the `password:` value and email are masked. Turn **Credentials** OFF with **High-security**
   ON: the key-format secrets stay masked (mandatory), but the free-form password unmasks. Turn High-security OFF: everything unmasks.
4. Switch scene → **whatsapp**. Header shows *Gemma … inferring*; after ~10–20 s (CPU) the whole chat becomes
   "Private conversation hidden". Toggle **Private conversations** OFF → the chat is visible on the next frame; ON → hidden again.
5. Change **Style** to Pixelate/Blur: PII/chat masks change; credentials stay black.
6. Stop `ollama` (`pkill -x ollama`): header → "PROTECTED (Gemma offline: OCR/patterns only)", masks already placed stay.
7. Real screen: `python -m app.main` → show a terminal with `export TOKEN=<fake>`; select the Privacy Lens camera in Meet.

Offline renders: `python scripts/render_demo.py [--gemma]` writes before/after PNGs to `out/` (the "before" images contain only fake data).
