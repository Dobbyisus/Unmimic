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
    serve.add_argument("--upstream", help="upstream model base URL (default: UNMIMIC_UPSTREAM_URL)")
    serve.add_argument("--host", help="interface to listen on (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port to listen on (default: 9000)")

    sub.add_parser("audit", help="measure how distillable the model's outputs are")

    return parser


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from unmimic.config import Settings
    from unmimic.proxy.app import create_app

    settings = Settings()
    upstream = args.upstream or settings.upstream_url
    host = args.host or settings.host
    port = args.port or settings.port
    app = create_app(upstream, upstream_api_key=settings.upstream_api_key)
    print(f"Unmimic proxy on http://{host}:{port}/v1 -> {upstream}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
    return 0


COMMANDS = {"serve": serve}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = COMMANDS.get(args.command)
    if command is None:
        print(f"unmimic {args.command}: not implemented yet", file=sys.stderr)
        return NOT_IMPLEMENTED
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
