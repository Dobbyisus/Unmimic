"""Build the normal-prompt sets from WildChat-1M (Ai2, ODC-BY license; credit Ai2).

Downloads a few parquet shards (~450 MB), keeps the first user message of each non-toxic
conversation, drops prompts that look like reasoning-extraction attempts (real users tried
jailbreaks too), deduplicates, and writes two disjoint files:

    data/benign/wildchat_train.jsonl  - the input guard learns "normal" from these
    data/benign/wildchat_eval.jsonl   - held back to measure false positives

Run with ``unmimic fetch-benign``. Needs the ``ml`` extras.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = "allenai/WildChat-1M"
# Spread over the collection period, so prompts aren't all from one month.
DEFAULT_SHARDS = ("data/train-00000-of-00014.parquet", "data/train-00009-of-00014.parquet")

# Prompts that mention the model's reasoning together with a reveal-style verb. Dropped
# from the normal set so attack-like prompts are never taught as "normal".
EXTRACTION_LIKE = re.compile(
    r"(reasoning|chain[- ]of[- ]thought|thought process|thinking|internal|hidden|"
    r"system prompt|instructions above|your prompt).{0,80}"
    r"(repeat|reveal|print|show|output|verbatim|word for word|translate|encode|base64)|"
    r"(repeat|reveal|print|show|output|verbatim|word for word).{0,80}"
    r"(reasoning|chain[- ]of[- ]thought|thought process|internal|hidden|system prompt)",
    re.IGNORECASE | re.DOTALL,
)


def first_user_prompts(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["conversation", "toxic", "redacted"])
    prompts = []
    columns = [table[c].to_pylist() for c in table.column_names]
    for conversation, toxic, redacted in zip(*columns, strict=True):
        if toxic or redacted or not conversation:
            continue
        first = conversation[0]
        if first.get("role") == "user":
            prompts.append((first.get("content") or "").strip())
    return prompts


def keep(prompt: str) -> bool:
    return 3 <= len(prompt) <= 2000 and not EXTRACTION_LIKE.search(prompt)


def fetch(
    out: Path = Path("data/benign"),
    train: int = 10_000,
    held_back: int = 10_000,
    shards: tuple[str, ...] = DEFAULT_SHARDS,
    seed: int = 0,
) -> list[Path]:
    """Download, filter and split; returns the files written (train, then held back)."""
    from huggingface_hub import hf_hub_download

    seen: set[str] = set()
    prompts: list[str] = []
    dropped = 0
    for shard in shards:
        path = Path(hf_hub_download(REPO, shard, repo_type="dataset"))
        for prompt in first_user_prompts(path):
            key = " ".join(prompt.lower().split())
            if key in seen:
                continue
            seen.add(key)
            if keep(prompt):
                prompts.append(prompt)
            elif EXTRACTION_LIKE.search(prompt):
                dropped += 1

    need = train + held_back
    if len(prompts) < need:
        raise ValueError(f"only {len(prompts)} prompts after filtering; need {need}")
    random.Random(seed).shuffle(prompts)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, chunk in [("train", prompts[:train]), ("eval", prompts[train:need])]:
        path = out / f"wildchat_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for prompt in chunk:
                f.write(json.dumps({"prompt": prompt, "source": REPO}, ensure_ascii=False) + "\n")
        print(f"wrote {len(chunk)} prompts to {path}")
        written.append(path)
    print(f"{len(prompts)} usable prompts; {dropped} dropped as extraction-like")
    return written
