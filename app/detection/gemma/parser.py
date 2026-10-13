"""Validate untrusted model output before it reaches the policy/redaction layers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.schema import Box, Category, Detection

CATEGORY_ALIASES = {
    "credentials": Category.CREDENTIALS, "credential": Category.CREDENTIALS, "password": Category.CREDENTIALS,
    "secret": Category.CREDENTIALS, "api_key": Category.CREDENTIALS,
    "private_chat": Category.PRIVATE_CHAT, "chat": Category.PRIVATE_CHAT, "messaging": Category.PRIVATE_CHAT,
    "private_conversation": Category.PRIVATE_CHAT,
    "pii": Category.PII, "personal": Category.PII,
    "financial": Category.FINANCIAL, "finance": Category.FINANCIAL,
    "sensitive_document": Category.SENSITIVE_DOCUMENT, "document": Category.SENSITIVE_DOCUMENT,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "app": {"type": "string"},
        "category": {"type": "string", "enum": ["credentials", "private_chat", "pii", "financial", "sensitive_document", "none"]},
        "sensitive": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "regions": {"type": "array", "items": {
            "type": "object",
            "properties": {"category": {"type": "string"}, "x0": {"type": "number"}, "y0": {"type": "number"},
                           "x1": {"type": "number"}, "y1": {"type": "number"}},
            "required": ["category", "x0", "y0", "x1", "y1"]}},
    },
    "required": ["category", "sensitive", "confidence", "regions"],
}


@dataclass
class GemmaResult:
    app: str = ""
    category: Category | None = None
    sensitive: bool = False
    confidence: float = 0.0
    reason: str = ""
    regions: list[tuple[Category, Box | None]] = field(default_factory=list)  # Box None => sensitive but unlocalized
    warnings: list[str] = field(default_factory=list)


class GemmaParseError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)  # tolerate chatter around the object
        if not m:
            raise GemmaParseError("no JSON object in model output")
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError as e:
            raise GemmaParseError(f"malformed JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise GemmaParseError("JSON root is not an object")
    return obj


def _norm_coord_set(vals: list[float]) -> list[float]:
    """Accept normalized 0..1, 0..100 (percent) or 0..1000 (Gemma-style) coordinates."""
    m = max(vals)
    if m <= 1.5:
        return vals
    if m <= 100:
        return [v / 100 for v in vals]
    return [v / 1000 for v in vals]


def parse_response(text: str) -> GemmaResult:
    obj = _extract_json(text)
    res = GemmaResult(app=str(obj.get("app", ""))[:60], reason=str(obj.get("reason", ""))[:200])
    try:
        res.confidence = min(1.0, max(0.0, float(obj.get("confidence", 0))))
    except (TypeError, ValueError):
        res.confidence = 0.0
        res.warnings.append("bad confidence")
    sens = obj.get("sensitive", False)
    res.sensitive = sens if isinstance(sens, bool) else str(sens).lower() == "true"
    res.category = CATEGORY_ALIASES.get(str(obj.get("category", "")).lower().strip())
    regions = obj.get("regions") or []
    if not isinstance(regions, list):
        regions = []
        res.warnings.append("regions not a list")
    for r in regions[:12]:
        if not isinstance(r, dict):
            res.warnings.append("region not an object")
            continue
        cat = CATEGORY_ALIASES.get(str(r.get("category", obj.get("category", ""))).lower().strip()) or res.category
        if cat is None:
            res.warnings.append("region without category")
            continue
        try:
            v = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError):
            res.regions.append((cat, None))
            res.warnings.append("region missing coordinates")
            continue
        box = Box.validated(*_norm_coord_set(v))
        if box is None:
            res.warnings.append("region coordinates invalid")
        res.regions.append((cat, box))
    if res.sensitive and res.category and not res.regions:
        res.regions.append((res.category, None))
    return res


def to_detections(res: GemmaResult, min_localized_area: float = 0.01, fallback_box: Box | None = None) -> list[Detection]:
    """Turn a validated result into detections. Sensitive-but-unlocalized => conservative fallback box."""
    if not res.sensitive:
        return []
    fb = fallback_box or Box(0.0, 0.04, 1.0, 1.0)  # whole content area, minus a thin top strip
    out = []
    for cat, box in res.regions:
        localized = box is not None and box.area >= min_localized_area
        conf = res.confidence if localized else res.confidence * 0.9
        out.append(Detection(cat, conf, box if localized else fb, "gemma", "model_localized" if localized else "model_fallback_fullscreen"))
    return out
