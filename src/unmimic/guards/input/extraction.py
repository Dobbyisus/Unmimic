"""Extraction guard: the input guard that blocks reasoning-extraction prompts.

Stage 1 (logistic regression on embeddings) scores every request. Requests it flags go to
stage 2, the attack memory: near a known attack means block; otherwise the request is
allowed but flagged, so the account monitor can count it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from unmimic.corpus import Family, load_families, load_prompts
from unmimic.embeddings import embed, embed_cached
from unmimic.guards.input.memory import AttackMemory
from unmimic.guards.input.stage1 import Stage1
from unmimic.plugins import Context, Verdict, registry

# Defaults until per-deployment calibration; chosen with experiments/eval_input_guard.py
# (leave-one-type-out, 267 attacks of 11 types): at 0.5 / 0.70, 92% of attacks of an
# unseen type are blocked or flagged and 38% blocked; the deployed guard blocked 0.04% of
# 10k held-back WildChat prompts (95% upper bound 0.09%) and flagged 1.4%.
FLAG_THRESHOLD = 0.5
MEMORY_THRESHOLD = 0.70
CHUNK_WORDS = 40
MAX_CHUNKS = 16
SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def user_text(request: dict[str, Any]) -> str:
    """Text of the last user message (plain string or OpenAI content parts)."""
    for message in reversed(request.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "\n".join(p for p in parts if p)
        return ""
    return ""


def chunks(text: str) -> list[str]:
    """One chunk per sentence; sentences over ``CHUNK_WORDS`` words are cut into windows.

    Scoring sentences separately keeps an attack sentence from being diluted by the normal
    text around it. Long messages keep their first 4 and last 12 chunks.
    """
    pieces = []
    for sentence in SENTENCE_BREAK.split(text):
        words = sentence.split()
        pieces += [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    if len(pieces) > MAX_CHUNKS:
        pieces = pieces[:4] + pieces[-(MAX_CHUNKS - 4) :]
    return pieces


@registry.register("input", "extraction_guard")
class ExtractionGuard:
    """Blocks requests close to known reasoning-extraction attacks.

    With no arguments it trains on the folders and files in ``Settings``
    (``attack_dirs``, ``benign_files``). Embeddings of those corpora are cached on disk.
    """

    name = "extraction_guard"

    def __init__(
        self,
        families: list[Family] | None = None,
        benign: list[str] | None = None,
        flag_threshold: float = FLAG_THRESHOLD,
        memory_threshold: float = MEMORY_THRESHOLD,
        cache_dir: Path | None = None,
    ) -> None:
        if families is None or benign is None or cache_dir is None:
            from unmimic.config import Settings

            settings = Settings()
            families = load_families(*settings.attack_dirs) if families is None else families
            benign = load_prompts(*settings.benign_files) if benign is None else benign
            cache_dir = settings.data_dir / "cache" if cache_dir is None else cache_dir
        if not families:
            raise ValueError("no attack families found")
        if not benign:
            raise ValueError("no normal prompts found; run: unmimic fetch-benign")

        attacks = [example for family in families for example in family.examples]
        labels = [family.name for family in families for _ in family.examples]
        attack_vectors = embed_cached(attacks, cache_dir)
        benign_vectors = embed_cached([c for prompt in benign for c in chunks(prompt)], cache_dir)
        self.stage1 = Stage1().fit(attack_vectors, benign_vectors)
        self.memory = AttackMemory(attack_vectors, labels, benign_vectors)
        self.flag_threshold = flag_threshold
        self.memory_threshold = memory_threshold

    def check(self, request: dict[str, Any], ctx: Context) -> Verdict:
        pieces = chunks(user_text(request))
        if not pieces:
            return Verdict(action="allow", score=0.0)
        return self.decide(embed(pieces))

    def decide(self, vectors: Any) -> Verdict:
        """Verdict for one request, given the embeddings of its chunks."""
        scores = self.stage1.score(vectors)
        top = float(scores.max())
        if top < self.flag_threshold:
            return Verdict(action="allow", score=top)
        flagged = vectors[scores >= self.flag_threshold]
        best = max(self.memory.nearest(flagged), key=lambda m: m.attack_similarity)
        near = f"{best.attack_family} ({best.attack_similarity:.2f})"
        if (
            best.attack_similarity >= self.memory_threshold
            and best.attack_similarity > best.benign_similarity
        ):
            return Verdict(action="block", score=top, reason=f"near known attack {near}")
        return Verdict(action="flag", score=top, reason=f"stage 1 only; nearest attack {near}")
