"""Decoders tried on an answer before comparing it with the reasoning.

Attackers ask the model to hide its reasoning in the answer: base64, ROT13, reversed
text, or wrapped in tags. Each decoder returns the decoded text, or None when it does not
apply.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re

# Runs of base64 characters, allowing line wraps inside a run.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/=\r\n]{16,}")
MARKUP = re.compile(r"<[^<>\n]{1,40}>|```[\w-]*")


def is_readable(text: str) -> bool:
    if len(text) < 4 or not any(c.isalpha() for c in text):
        return False
    printable = sum(c.isprintable() or c.isspace() for c in text)
    return printable / len(text) >= 0.95


def decode_base64(text: str) -> str | None:
    pieces = []
    for match in BASE64_RUN.finditer(text):
        run = re.sub(r"\s+", "", match.group())
        try:
            decoded = base64.b64decode(run + "=" * (-len(run) % 4)).decode("utf-8")
        except (binascii.Error, ValueError):  # UnicodeDecodeError is a ValueError
            continue
        if is_readable(decoded):
            pieces.append(decoded)
    return " ".join(pieces) or None


def decode_rot13(text: str) -> str:
    return codecs.decode(text, "rot13")


def reverse(text: str) -> str:
    return text[::-1]


def strip_markup(text: str) -> str | None:
    stripped = MARKUP.sub(" ", text)
    return stripped if stripped != text else None


def decode_variants(text: str) -> list[tuple[str, str]]:
    """``(name, text)`` for the answer as written and every decoding that applies."""
    variants = [
        ("plain", text),
        ("markup", strip_markup(text)),
        ("base64", decode_base64(text)),
        ("rot13", decode_rot13(text)),
        ("reversed", reverse(text)),
    ]
    return [(name, value) for name, value in variants if value]
