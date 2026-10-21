"""Where a model puts its reasoning, and how to split it from the answer.

``unmimic init`` writes one ``ModelFormat`` per model (see ``adapt/probe.py``); the proxy
reads them so output guards always receive the reasoning as plain text, whatever the
model's format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

FormatKind = Literal["field", "tag", "marker", "none"]

# API fields that servers use for separated reasoning (vLLM, Ollama, ...).
KNOWN_FIELDS = ("reasoning_content", "reasoning")


class ModelFormat(BaseModel):
    """Where one model's reasoning appears.

    - ``field``: a separate API field (``field``, or any of ``KNOWN_FIELDS`` if unset).
    - ``tag``: inside ``open_tag`` ... ``close_tag`` in the text. ``implicit_open`` means the
      chat template adds the opening tag, so the output starts mid-reasoning.
    - ``marker``: the text before ``marker`` (e.g. "Final Answer:").
    - ``none``: no visible reasoning.
    """

    model: str = ""
    kind: FormatKind = "field"
    field: str | None = None
    open_tag: str | None = None
    close_tag: str | None = None
    implicit_open: bool = False
    marker: str | None = None
    # How many probe replies were seen, and how many agreed on this format.
    samples: int = 0
    matches: int = 0

    def save(self, models_dir: Path) -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{safe_filename(self.model)}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ModelFormat:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


DEFAULT_FORMAT = ModelFormat()


def safe_filename(model: str) -> str:
    """Model names like ``qwen3:8b`` or ``org/model`` are not valid Windows filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model) or "model"


def load_formats(models_dir: Path) -> dict[str, ModelFormat]:
    """All saved formats, keyed by model name."""
    if not models_dir.is_dir():
        return {}
    formats = (ModelFormat.load(path) for path in sorted(models_dir.glob("*.json")))
    return {fmt.model: fmt for fmt in formats}


@dataclass(frozen=True)
class Split:
    reasoning: str
    answer: str


def split_text(text: str, fmt: ModelFormat) -> Split:
    """Split reply text for the ``tag`` and ``marker`` formats; other formats return it all
    as the answer."""
    if fmt.kind == "tag" and fmt.close_tag:
        return _split_tags(text, fmt.open_tag or "", fmt.close_tag, fmt.implicit_open)
    if fmt.kind == "marker" and fmt.marker:
        at = text.find(fmt.marker)
        if at < 0:
            return Split("", text)
        return Split(text[:at].strip(), text[at:].strip())
    return Split("", text)


def _split_tags(text: str, open_tag: str, close_tag: str, implicit_open: bool) -> Split:
    start = text.find(open_tag) if open_tag else -1
    close = text.find(close_tag)
    if start >= 0 and (close < 0 or start < close):
        prefix, body = text[:start], start + len(open_tag)
    elif implicit_open or close >= 0:
        # A closing tag with no opening tag before it: everything before it is reasoning
        # (a skipped opening tag, or one the model wrote later in its answer).
        prefix, body = "", 0
    else:
        return Split("", text)
    # The LAST closing tag ends the reasoning. Asked about their thinking, models sometimes
    # close it, write a second block of reasoning and close again; cutting at the first
    # tag shipped that second block as answer (Experiment 1, DeepSeek-R1: 4 of 267 replies
    # through Unmimic). A closing tag inside a genuine answer is rare, and hiding more is
    # the safe failure.
    end = text.rfind(close_tag, body)
    if end < 0:  # still thinking, or cut off mid-reasoning
        return Split(text[body:].strip(), prefix.strip())
    return Split(text[body:end].strip(), (prefix + text[end + len(close_tag) :]).strip())


def _field_value(part: dict[str, Any], fmt: ModelFormat) -> str:
    fields = (fmt.field,) if fmt.field else KNOWN_FIELDS
    return next((part[f] for f in fields if part.get(f)), "")


def split_message(message: dict[str, Any], fmt: ModelFormat) -> Split:
    """Split a complete chat message (``choices[i].message``)."""
    content = message.get("content") or ""
    if fmt.kind == "field":
        return Split(_field_value(message, fmt), content)
    return split_text(content, fmt)


class StreamReasoning:
    """Follows one streamed choice; ``reasoning`` is all reasoning recognised so far.

    Text formats are re-split on the accumulated text, so tags split across chunks
    (``<thi`` + ``nk>``) are still found.
    """

    def __init__(self, fmt: ModelFormat) -> None:
        self.fmt = fmt
        self._field = ""
        self._text = ""

    def feed(self, delta: dict[str, Any]) -> None:
        if self.fmt.kind == "field":
            self._field += _field_value(delta, self.fmt)
        else:
            self._text += delta.get("content") or ""

    @property
    def reasoning(self) -> str:
        if self.fmt.kind == "field":
            return self._field
        return split_text(self._text, self.fmt).reasoning
