"""Build the normal-prompt sets from WildChat-1M (Ai2, ODC-BY license; credit Ai2).

Downloads a few parquet shards, keeps the first user message of each non-toxic
conversation, drops prompts that look like reasoning-extraction attempts (real users
tried jailbreaks too), deduplicates, and writes two disjoint files:

    data/benign/wildchat_train.jsonl  - the input guard learns "normal" from these
    data/benign/wildchat_eval.jsonl   - held back to measure false positives

Runs on a laptop: python experiments/fetch_wildchat.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = "allenai/WildChat-1M"
# Spread over the collection period, so prompts aren't all from one month.
DEFAULT_SHARDS = ["data/train-00000-of-00014.parquet", "data/train-00009-of-00014.parquet"]

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--eval", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("data/benign"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    seen: set[str] = set()
    prompts: list[str] = []
    dropped = 0
    for shard in args.shards:
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

    need = args.train + args.eval
    if len(prompts) < need:
        raise SystemExit(f"only {len(prompts)} prompts after filtering; need {need}")
    random.Random(args.seed).shuffle(prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, chunk in [("train", prompts[: args.train]), ("eval", prompts[args.train : need])]:
        path = args.out / f"wildchat_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for prompt in chunk:
                f.write(json.dumps({"prompt": prompt, "source": REPO}, ensure_ascii=False) + "\n")
        print(f"wrote {len(chunk)} prompts to {path}")
    print(f"{len(prompts)} usable prompts; {dropped} dropped as extraction-like")


if __name__ == "__main__":
    main()
