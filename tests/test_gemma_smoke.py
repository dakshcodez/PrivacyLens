"""Real local inference. Run with: pytest -m smoke -o addopts=''  (needs ollama serve + the model pulled)."""
import pytest

from app.capture.synthetic import SCENES
from app.detection.gemma.client import GemmaDetector
from app.main import load_config
from app.schema import Category

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def det():
    g = load_config()["gemma"]
    d = GemmaDetector(g["host"], g["model"], g["image_max_side"], g["timeout_s"], g["num_predict"], g["temperature"])
    ok, msg = d.available()
    if not ok:
        pytest.skip(msg)
    return d


def test_whatsapp_is_private_chat(det):
    r = det.classify(SCENES["whatsapp"]())
    assert r.sensitive and r.category == Category.PRIVATE_CHAT


def test_plain_document_not_sensitive(det):
    assert not det.classify(SCENES["document"]()).sensitive
