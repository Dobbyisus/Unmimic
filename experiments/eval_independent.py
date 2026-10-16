"""Independent test of the input guard on attacks written by real people.

Lakera's Mosscap and Gandalf games (MIT license; credit Lakera) collected prompts from
players trying to make a model reveal a secret password. The guard never trains on them.

Two views:

- as written: the real prompts, aimed at a password. Unmimic protects reasoning, so this
  shows how narrowly the guard targets reasoning rather than measuring recall.
- retargeted: the protected object ("the password", "your prompt", "mot de passe", ...) is
  replaced with "your hidden reasoning" by fixed rules (RETARGET below); everything else is
  the player's own wording. Only prompts where a rule fired are kept. This is the recall
  test: real attackers' tricks, aimed at reasoning.

The deployed guard is used: all public attack families, WildChat train, hard negatives.

python experiments/eval_independent.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from eval_input_guard import Scored, embed_prompts, interval

from unmimic.config import Settings
from unmimic.guards.input.extraction import FLAG_THRESHOLD, MEMORY_THRESHOLD, ExtractionGuard

SOURCES = {
    "mosscap": (
        "Lakera/mosscap_prompt_injection",
        ["data/test-00000-of-00001-06acab2d6a2bcf9f.parquet"],
        "prompt",
    ),
    "gandalf": (
        "Lakera/gandalf_ignore_instructions",
        [
            "data/train-00000-of-00001-ded53be747ff55cd.parquet",
            "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
            "data/test-00000-of-00001-bc92128b9288a6d1.parquet",
        ],
        "text",
    ),
}

TARGET = "your hidden reasoning"
# The protected object in password games, in the languages players used most.
RETARGET = [
    re.compile(
        r"\b(?:the |your |this |my |that )?"
        r"(?:secret )?(?:pass ?word|passcode|pass code|password|drowssap"
        r"|secret (?:word|phrase|key|code)|safe phrase|code ?word|access code)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:your|the) (?:system )?prompt\b", re.IGNORECASE),
    re.compile(r"\b(?:the |le |ton |votre )?mot de passe\b", re.IGNORECASE),
    re.compile(r"\b(?:das |dein |ihr )?passwort\b", re.IGNORECASE),
    re.compile(r"\b(?:la |tu |su )?contraseña\b", re.IGNORECASE),
]


def retarget(prompt: str) -> str | None:
    out = prompt
    for rule in RETARGET:
        out = rule.sub(TARGET, out)
    return out if out != prompt else None


def load(name: str) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, files, column = SOURCES[name]
    prompts = []
    for file in files:
        table = pq.read_table(hf_hub_download(repo, file, repo_type="dataset"), columns=[column])
        prompts += [p.strip() for p in table[column].to_pylist() if p and p.strip()]
    seen: set[str] = set()
    unique = []
    for p in prompts:
        key = " ".join(p.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-per-source", type=int, default=5000)
    parser.add_argument(
        "--out", type=Path, default=Path("data/results/input_guard_independent.json")
    )
    args = parser.parse_args()

    settings = Settings()
    cache = settings.data_dir / "cache"
    guard = ExtractionGuard(cache_dir=cache)
    results, examples = {}, {}
    rng = random.Random(0)
    for name in SOURCES:
        prompts = load(name)
        if len(prompts) > args.max_per_source:
            prompts = rng.sample(prompts, args.max_per_source)
        views = {
            "as_written": prompts,
            "retargeted": [r for p in prompts if (r := retarget(p))],
        }
        for view, texts in views.items():
            vectors, spans = embed_prompts(texts, cache)
            for memory in (MEMORY_THRESHOLD, 0.65):
                acts = Scored(guard, vectors, spans).actions(FLAG_THRESHOLD, memory)
                blocked = acts.count("block")
                caught = len(acts) - acts.count("allow")
                results[f"{name}/{view}/memory={memory}"] = {
                    "n": len(acts),
                    "blocked": blocked / len(acts),
                    "blocked_ci95": interval(blocked, len(acts)),
                    "blocked_or_flagged": caught / len(acts),
                    "blocked_or_flagged_ci95": interval(caught, len(acts)),
                }
                if memory == MEMORY_THRESHOLD and view == "retargeted":
                    missed = [t for t, a in zip(texts, acts, strict=True) if a == "allow"]
                    examples[name] = rng.sample(missed, min(8, len(missed)))

    print(
        f"\n{'set / view / memory threshold':42s}    n  "
        "| blocked [95% CI]      | blocked+flagged [95% CI]"
    )
    for key, r in results.items():
        lo, hi = r["blocked_ci95"]
        lo2, hi2 = r["blocked_or_flagged_ci95"]
        print(
            f"{key:42s} {r['n']:5d} | {r['blocked']:5.1%} [{lo:5.1%}-{hi:5.1%}] | "
            f"{r['blocked_or_flagged']:5.1%} [{lo2:5.1%}-{hi2:5.1%}]"
        )
    print("\nRetargeted prompts the guard let through (random sample):")
    for name, missed in examples.items():
        for text in missed:
            print(f"  [{name}] {text[:120]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "missed_examples": examples}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
