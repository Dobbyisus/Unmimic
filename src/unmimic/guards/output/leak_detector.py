"""Leak detector: does the answer contain the model's hidden reasoning?

Checks, cheapest first:

1. Exact: the share of the reasoning's 8-word sequences that also appear in the answer,
   tried on the answer as written and on every decoding (base64, ROT13, reversed, markup
   stripped).
2. Meaning: the share of reasoning sentences with a close match in the answer, using a
   multilingual embedding model, so translated copies are caught. Needs the ``ml`` extras
   and runs only when the exact check finds little.

``LeakGuard`` is the output guard: it replaces any answer that leaks with a notice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from unmimic.guards.output.decoders import decode_base64, decode_variants
from unmimic.plugins import Context, registry
from unmimic.reasoning import DEFAULT_FORMAT, ModelFormat, split_text

log = logging.getLogger(__name__)

NGRAM = 8
# An exact score this high is conclusive, so the slower meaning check is skipped.
EXACT_CUTOFF = 0.3
DEFAULT_THRESHOLD = 0.3
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cosine similarity at which two sentences count as the same statement. Measured on the
# default model: translations score 0.8-0.99, close paraphrases 0.75-0.85.
SIMILAR = 0.75
# Honest answers restate one reasoning sentence (usually the conclusion) all the time, so
# fewer matches than this scores 0.
MIN_MATCHED_SENTENCES = 2
NOTICE = "This response was withheld by Unmimic."
# While streaming, re-check after this many new answer characters (and at the end).
STREAM_CHECK_CHARS = 80

WORD = re.compile(r"\w+")
SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(answer: str, reasoning: str, n: int = NGRAM) -> float:
    """Share of the reasoning's n-word sequences found in the answer.

    Reasoning shorter than n words scores 0: too short to tell a leak from a coincidence.
    """
    target = ngrams(words(reasoning), n)
    if not target:
        return 0.0
    return len(target & ngrams(words(answer), n)) / len(target)


def sentences(text: str, min_words: int = 3) -> list[str]:
    return [s.strip() for s in SENTENCE_BREAK.split(text) if len(words(s)) >= min_words]


class SemanticMatcher:
    """Meaning overlap using a multilingual sentence-embedding model."""

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._cached: tuple[str, Any] | None = None

    def _reasoning_embeddings(self, reasoning: str) -> Any:
        # Streaming re-checks the same reasoning many times; embed it once.
        if self._cached is None or self._cached[0] != reasoning:
            targets = sentences(reasoning)
            embedded = self.model.encode(targets, normalize_embeddings=True) if targets else None
            self._cached = (reasoning, embedded)
        return self._cached[1]

    def overlap(self, answer: str, reasoning: str) -> float:
        targets = self._reasoning_embeddings(reasoning)
        found = sentences(answer)
        if targets is None or not found:
            return 0.0
        candidates = self.model.encode(found, normalize_embeddings=True)
        matched = int(((targets @ candidates.T).max(axis=1) >= SIMILAR).sum())
        if matched < MIN_MATCHED_SENTENCES:
            return 0.0
        return matched / len(targets)


def load_matcher(model_name: str) -> SemanticMatcher | None:
    try:
        return SemanticMatcher(model_name)
    except ImportError:
        log.warning("leak detector: meaning check off; install with: pip install -e '.[ml]'")
        return None


@dataclass(frozen=True)
class Leak:
    score: float
    check: str  # "exact", "meaning", or "none"
    variant: str  # which decoding of the answer matched: "plain", "base64", ...


def leak_score(answer: str, reasoning: str, matcher: SemanticMatcher | None = None) -> Leak:
    """How much of the reasoning appears in the answer, from 0 to 1."""
    if not answer.strip() or not reasoning.strip():
        return Leak(0.0, "none", "plain")
    best = max(
        (
            Leak(ngram_overlap(text, reasoning), "exact", name)
            for name, text in decode_variants(answer)
        ),
        key=lambda leak: leak.score,
    )
    if best.score >= EXACT_CUTOFF or matcher is None:
        return best
    for name, text in [("plain", answer), ("base64", decode_base64(answer))]:
        if text:
            meaning = Leak(matcher.overlap(text, reasoning), "meaning", name)
            if meaning.score > best.score:
                best = meaning
    return best


@registry.register("output", "leak_detector")
class LeakGuard:
    """Replaces any answer that contains the hidden reasoning with a notice.

    Streaming: the answer is re-checked as it grows. Once a leak is found the rest of the
    stream is blanked; text sent before that point cannot be recalled.
    """

    name = "leak_detector"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        semantic_model: str | None = DEFAULT_SEMANTIC_MODEL,
    ) -> None:
        self.threshold = threshold
        self.matcher = load_matcher(semantic_model) if semantic_model else None

    def transform(self, response: dict[str, Any], reasoning: str, ctx: Context) -> dict[str, Any]:
        fmt: ModelFormat = ctx.metadata.get("format", DEFAULT_FORMAT)
        if ctx.metadata.get("stream"):
            return self._transform_chunk(response, reasoning, ctx, fmt)
        for choice in response.get("choices", []):
            message = choice.get("message") or {}
            answer = split_text(message.get("content") or "", fmt).answer
            leak = leak_score(answer, reasoning, self.matcher)
            if leak.score >= self.threshold:
                self._record(ctx, leak)
                message["content"] = NOTICE
                choice["finish_reason"] = "content_filter"
        return response

    def _transform_chunk(
        self, chunk: dict[str, Any], reasoning: str, ctx: Context, fmt: ModelFormat
    ) -> dict[str, Any]:
        state = ctx.metadata.setdefault(self.name, {"text": {}, "checked": {}, "tripped": False})
        for choice in chunk.get("choices", []):
            delta = choice.setdefault("delta", {})
            finished = choice.get("finish_reason") is not None
            if state["tripped"]:
                if delta.get("content"):
                    delta["content"] = ""
                if finished:
                    choice["finish_reason"] = "content_filter"
                continue

            index = choice.get("index", 0)
            state["text"][index] = state["text"].get(index, "") + (delta.get("content") or "")
            answer = split_text(state["text"][index], fmt).answer
            grown = len(answer) - state["checked"].get(index, 0)
            if not finished and grown < STREAM_CHECK_CHARS:
                continue
            state["checked"][index] = len(answer)
            leak = leak_score(answer, reasoning, self.matcher)
            if leak.score >= self.threshold:
                state["tripped"] = True
                self._record(ctx, leak)
                delta["content"] = "\n\n" + NOTICE
                if finished:
                    choice["finish_reason"] = "content_filter"
        return chunk

    def _record(self, ctx: Context, leak: Leak) -> None:
        event = {
            "guard": self.name,
            "score": round(leak.score, 3),
            "check": leak.check,
            "variant": leak.variant,
        }
        ctx.metadata.setdefault("leaks", []).append(event)
        log.warning(
            "leak withheld: account=%s model=%s score=%.2f check=%s variant=%s",
            ctx.account_id,
            ctx.model,
            leak.score,
            leak.check,
            leak.variant,
        )
