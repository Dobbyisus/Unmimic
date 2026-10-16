"""Command-line interface: ``unmimic init | redteam | calibrate | serve | audit``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from unmimic import __version__

NOT_IMPLEMENTED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unmimic",
        description="Protect an LLM served over an OpenAI-compatible API from distillation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    init = sub.add_parser("init", help="probe the model and write its format config")
    init.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")

    redteam = sub.add_parser("redteam", help="attack the model and record what leaks")
    redteam.add_argument("--rounds", type=int, default=3)

    calibrate = sub.add_parser("calibrate", help="set thresholds for a target false-positive rate")
    calibrate.add_argument("--benign", required=True, help="JSONL file of normal prompts")
    calibrate.add_argument("--fpr", type=float, default=0.001)

    serve = sub.add_parser("serve", help="run the protecting proxy")
    serve.add_argument("--port", type=int, default=9000)
    serve.add_argument("--upstream", help="upstream model base URL")

    sub.add_parser("audit", help="measure how distillable the model's outputs are")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"unmimic {args.command}: not implemented yet", file=sys.stderr)
    return NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
