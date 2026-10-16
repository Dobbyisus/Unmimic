"""Strip transform: remove the model's reasoning before a reply leaves the proxy.

Removes reasoning API fields (``reasoning_content``, ``reasoning``, or the model's own
field) and, for text formats, everything but the answer: ``<think>`` blocks, or the text
before a "Final Answer:" marker.

Run it after the leak detector, which needs the full text to tell answer from reasoning.
"""

from __future__ import annotations

from typing import Any

from unmimic.plugins import Context, registry
from unmimic.reasoning import DEFAULT_FORMAT, KNOWN_FIELDS, ModelFormat, split_text


def reasoning_fields(fmt: ModelFormat) -> set[str]:
    return set(KNOWN_FIELDS) | ({fmt.field} if fmt.field else set())


def hold_back(text: str, fmt: ModelFormat) -> str:
    """Drop a trailing piece that may be the start of a tag or marker still arriving."""
    for token in (fmt.open_tag, fmt.close_tag, fmt.marker):
        if not token:
            continue
        for size in range(len(token) - 1, 0, -1):
            if text.endswith(token[:size]):
                return text[:-size]
    return text


def visible_answer(text: str, fmt: ModelFormat, final: bool) -> str:
    """The part of a (possibly unfinished) reply that is safe to show."""
    if fmt.kind not in ("tag", "marker"):
        return text
    if not final:
        text = hold_back(text, fmt)
        if fmt.kind == "marker" and fmt.marker not in text:
            return ""  # until the marker arrives, everything so far is reasoning
    return split_text(text, fmt).answer


@registry.register("output", "strip_reasoning")
class StripReasoning:
    """Removes reasoning from replies so users only receive the answer."""

    name = "strip_reasoning"

    def transform(self, response: dict[str, Any], reasoning: str, ctx: Context) -> dict[str, Any]:
        fmt: ModelFormat = ctx.metadata.get("format", DEFAULT_FORMAT)
        fields = reasoning_fields(fmt)
        if ctx.metadata.get("stream"):
            return self._transform_chunk(response, ctx, fmt, fields)
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            for field in fields:
                message.pop(field, None)
            # Another guard already replaced the content (e.g. the leak notice): keep it.
            if choice.get("finish_reason") == "content_filter":
                continue
            if fmt.kind in ("tag", "marker"):
                message["content"] = split_text(message.get("content") or "", fmt).answer
        return response

    def _transform_chunk(
        self, chunk: dict[str, Any], ctx: Context, fmt: ModelFormat, fields: set[str]
    ) -> dict[str, Any]:
        state = ctx.metadata.setdefault(self.name, {"text": {}, "sent": {}})
        for choice in chunk.get("choices", []):
            delta = choice.get("delta")
            if delta is None:
                continue
            for field in fields:
                delta.pop(field, None)
            if fmt.kind not in ("tag", "marker"):
                continue

            # Re-split everything received so far and send only the new part of the answer.
            index = choice.get("index", 0)
            text = state["text"].get(index, "") + (delta.get("content") or "")
            state["text"][index] = text
            answer = visible_answer(text, fmt, final=choice.get("finish_reason") is not None)
            sent = state["sent"].get(index, "")
            new = answer[len(sent) :] if answer.startswith(sent) else ""
            state["sent"][index] = sent + new
            if new or "content" in delta:
                delta["content"] = new
        return chunk
