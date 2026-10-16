"""Held-out evaluation of the input guard.

Leave-one-family-out: for each attack family, train the guard on all the OTHER families and
test on this one, so every recall number is for a kind of attack the guard never saw.
False positives are measured on WildChat prompts kept out of training
(data/benign/wildchat_eval.jsonl), reported with a one-sided 95% Clopper-Pearson upper
bound. A threshold grid shows the trade-off.

python experiments/eval_input_guard.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from unmimic.config import Settings
from unmimic.corpus import load_families, load_prompts
from unmimic.embeddings import embed, embed_cached
from unmimic.guards.input.extraction import (
    FLAG_THRESHOLD,
    MEMORY_THRESHOLD,
    ExtractionGuard,
    chunks,
)

FLAG_GRID = [0.3, 0.5, 0.7, 0.9]
MEMORY_GRID = [0.65, 0.7, 0.75, 0.8]


def upper_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate."""
    return 1.0 if errors >= n else float(beta.ppf(confidence, errors + 1, n - errors))


def interval(hits: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval on a rate."""
    lo = 0.0 if hits == 0 else float(beta.ppf(0.025, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(0.975, hits + 1, n - hits))
    return lo, hi


class Scored:
    """Stage-1 scores and memory matches for every chunk of a set of prompts."""

    def __init__(self, guard: ExtractionGuard, vectors: np.ndarray, spans: list[tuple[int, int]]):
        self.spans = spans
        self.scores = guard.stage1.score(vectors)
        look = np.flatnonzero(self.scores >= min(FLAG_GRID))
        found = guard.memory.nearest(vectors[look]) if len(look) else []
        self.matches = dict(zip(look.tolist(), found, strict=True))

    def actions(self, flag: float, memory: float) -> list[str]:
        out = []
        for start, end in self.spans:
            flagged = [i for i in range(start, end) if self.scores[i] >= flag]
            if not flagged:
                out.append("allow")
                continue
            best = max((self.matches[i] for i in flagged), key=lambda m: m.attack_similarity)
            near = (
                best.attack_similarity >= memory and best.attack_similarity > best.benign_similarity
            )
            out.append("block" if near else "flag")
        return out


def embed_prompts(prompts: list[str], cache: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pieces, spans = [], []
    for prompt in prompts:
        parts = chunks(prompt) or [""]
        spans.append((len(pieces), len(pieces) + len(parts)))
        pieces += parts
    return embed_cached(pieces, cache), spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/results/input_guard_heldout.json"))
    args = parser.parse_args()

    settings = Settings()
    cache = settings.data_dir / "cache"
    families = load_families(*settings.attack_dirs)
    benign_train = load_prompts(*settings.benign_files)
    benign_eval = load_prompts(settings.data_dir / "benign" / "wildchat_eval.jsonl")
    print(
        f"{len(families)} families, {sum(len(f.examples) for f in families)} attacks; "
        f"{len(benign_train)} normal prompts for training, {len(benign_eval)} held back"
    )

    started = time.perf_counter()
    eval_vectors, eval_spans = embed_prompts(benign_eval, cache)
    print(f"embedded held-back normal prompts in {time.perf_counter() - started:.0f}s")

    grid = [(f, m) for f in FLAG_GRID for m in MEMORY_GRID]
    attack_actions: dict[tuple, list[str]] = {g: [] for g in grid}
    # Per fold: how many held-back normal prompts were blocked / flagged. Folds score the
    # same 10k prompts, so bounds use n = 10k per fold (not 10 x 10k).
    fold_blocked: dict[tuple, list[int]] = {g: [] for g in grid}
    fold_flagged: dict[tuple, list[int]] = {g: [] for g in grid}
    per_family = {}
    for held_out in families:
        train = [f for f in families if f.name != held_out.name]
        guard = ExtractionGuard(families=train, benign=benign_train, cache_dir=cache)
        test_vectors, test_spans = embed_prompts(list(held_out.examples), cache)
        on_attacks = Scored(guard, test_vectors, test_spans)
        on_benign = Scored(guard, eval_vectors, eval_spans)
        for g in grid:
            acts = on_attacks.actions(*g)
            attack_actions[g] += acts
            counts = on_benign.actions(*g)
            fold_blocked[g].append(counts.count("block"))
            fold_flagged[g].append(counts.count("flag"))
            if g == (FLAG_THRESHOLD, MEMORY_THRESHOLD):
                per_family[held_out.name] = {
                    a: acts.count(a) / len(acts) for a in ("block", "flag")
                }
        print(
            f"  held out {held_out.name:22s} blocked {per_family[held_out.name]['block']:.0%}"
            f"  flagged {per_family[held_out.name]['flag']:.0%}"
        )

    # The deployed guard learns every family; its false positives on held-back prompts.
    full = Scored(
        ExtractionGuard(families=families, benign=benign_train, cache_dir=cache),
        eval_vectors,
        eval_spans,
    )

    n = len(benign_eval)
    rows = []
    for g in grid:
        acts = attack_actions[g]
        caught = len(acts) - acts.count("allow")
        worst = max(fold_blocked[g])
        deployed = full.actions(*g)
        rows.append(
            {
                "flag_threshold": g[0],
                "memory_threshold": g[1],
                "n_attacks": len(acts),
                "attack_blocked": acts.count("block") / len(acts),
                "attack_blocked_ci95": interval(acts.count("block"), len(acts)),
                "attack_blocked_or_flagged": caught / len(acts),
                "attack_blocked_or_flagged_ci95": interval(caught, len(acts)),
                "n_benign": n,
                "heldout_benign_blocked_worst_fold": worst / n,
                "heldout_benign_blocked_worst_fold_upper95": upper_bound(worst, n),
                "heldout_benign_flagged_mean": float(np.mean(fold_flagged[g])) / n,
                "deployed_benign_blocked": deployed.count("block") / n,
                "deployed_benign_blocked_upper95": upper_bound(deployed.count("block"), n),
                "deployed_benign_flagged": deployed.count("flag") / n,
            }
        )
    print(
        f"\n{len(attack_actions[grid[0]])} held-out attacks; {n} held-back normal prompts\n"
        "flag  mem  | attacks blocked [95% CI]  | blocked+flagged [95% CI] "
        "| normal blocked, worst fold (<=95%) | deployed: blocked (<=95%), flagged"
    )
    for r in rows:
        lo, hi = r["attack_blocked_ci95"]
        lo2, hi2 = r["attack_blocked_or_flagged_ci95"]
        print(
            f"{r['flag_threshold']:.2f}  {r['memory_threshold']:.2f} | "
            f"{r['attack_blocked']:5.1%} [{lo:5.1%}-{hi:5.1%}] | "
            f"{r['attack_blocked_or_flagged']:5.1%} [{lo2:5.1%}-{hi2:5.1%}] | "
            f"{r['heldout_benign_blocked_worst_fold']:.2%} "
            f"(<= {r['heldout_benign_blocked_worst_fold_upper95']:.2%})            | "
            f"{r['deployed_benign_blocked']:.2%} "
            f"(<= {r['deployed_benign_blocked_upper95']:.2%}), "
            f"{r['deployed_benign_flagged']:.2%}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_family": per_family, "grid": rows}, indent=2))
    print(f"\nwrote {args.out}")
    _ = embed  # imported for interactive use


if __name__ == "__main__":
    main()
