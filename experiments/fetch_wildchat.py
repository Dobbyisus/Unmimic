"""Build the normal-prompt sets from WildChat-1M (Ai2, ODC-BY license; credit Ai2).

Same as ``unmimic fetch-benign`` (the logic lives in ``unmimic.benign``); kept so older
instructions still work:

    python experiments/fetch_wildchat.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from unmimic.benign import DEFAULT_SHARDS, fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", nargs="+", default=list(DEFAULT_SHARDS))
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--eval", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("data/benign"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fetch(args.out, args.train, args.eval, tuple(args.shards), args.seed)


if __name__ == "__main__":
    main()
