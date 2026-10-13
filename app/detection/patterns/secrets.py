"""Deterministic detectors for secrets/PII in plain text. Returns match *spans and labels only*."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.schema import Category


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    category: Category
    rule: str
    confidence: float
    mandatory: bool = False


# (rule, regex, category, confidence, mandatory)
_FORMATS = [
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", Category.CREDENTIALS, 0.99, True),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", Category.CREDENTIALS, 0.97, True),
    ("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b", Category.CREDENTIALS, 0.98, True),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", Category.CREDENTIALS, 0.97, True),
    ("stripe_key", r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b", Category.CREDENTIALS, 0.97, True),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b", Category.CREDENTIALS, 0.96, True),
    ("anthropic_openai_key", r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}\b", Category.CREDENTIALS, 0.95, True),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\b", Category.CREDENTIALS, 0.96, True),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}", Category.CREDENTIALS, 0.9, True),
    ("url_credentials", r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/@]{3,}@", Category.CREDENTIALS, 0.93, True),
    ("email", r"\b[A-Za-z0-9._%+\-]+ ?@ ?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", Category.PII, 0.9, False),
    ("phone", r"(?<![\w.])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]\d{3,4}[\s.\-]\d{3,4}(?![\w.])", Category.PII, 0.7, False),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", Category.PII, 0.85, False),
    ("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", Category.FINANCIAL, 0.8, False),
]
_FORMATS = [(n, re.compile(p), c, conf, m) for n, p, c, conf, m in _FORMATS]

# key = value assignments where the key names a secret; the *value* is what gets matched.
_ASSIGN = re.compile(
    r"(?i)\b(?:pass(?:word|wd)?|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|private[_\-]?key|auth|credential)s?\b"
    r"\s*[:=]\s*[\"']?([^\s\"']{4,})")
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_\-=]{24,}")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in (s.count(ch) for ch in set(s)))


def luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in digits][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(ds))
    return total % 10 == 0


def looks_like_secret_token(tok: str) -> bool:
    """Format-agnostic: long, mixed-class, high-entropy strings (not words, paths or hex hashes of plain words)."""
    if len(tok) < 20:
        return False
    classes = sum(bool(re.search(p, tok)) for p in (r"[a-z]", r"[A-Z]", r"\d"))
    if classes < 2:
        return False
    if tok.count("/") > 2 or tok.startswith(("http", "www")):
        return False
    return shannon_entropy(tok) >= 3.5


def find_matches(text: str) -> list[Match]:
    out: list[Match] = []
    for rule, rx, cat, conf, mand in _FORMATS:
        for m in rx.finditer(text):
            if rule == "phone" and len(re.sub(r"\D", "", m.group())) < 9:
                continue
            out.append(Match(m.start(), m.end(), cat, rule, conf, mand))
    for m in _ASSIGN.finditer(text):
        val = m.group(1)
        if val.lower() in {"null", "none", "true", "false", "required", "string", "hidden"}:
            continue
        out.append(Match(m.start(1), m.end(1), Category.CREDENTIALS, "assigned_secret", 0.85, True))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            out.append(Match(m.start(), m.end(), Category.FINANCIAL, "card_number", 0.92, True))
    for m in _LONG_TOKEN.finditer(text):
        if looks_like_secret_token(m.group()) and not any(o.start <= m.start() and m.end() <= o.end for o in out):
            out.append(Match(m.start(), m.end(), Category.CREDENTIALS, "high_entropy_token", 0.7, False))
    return out
