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
    init.add_argument("--model", help="model to probe (default: the only model served)")
    init.add_argument("--samples", type=int, default=20, help="questions to ask (max 20)")
    init.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="reply length limit; reasoning models need room to reach their closing tag",
    )

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


def init(args: argparse.Namespace) -> int:
    import httpx

    from unmimic.adapt.probe import ProbeError, describe, pick_model, probe
    from unmimic.config import Settings

    settings = Settings()
    key = settings.upstream_api_key
    headers = {"authorization": f"Bearer {key}"} if key else {}

    def progress(done: int, total: int) -> None:
        print(f"\r  reply {done}/{total}", end="", file=sys.stderr, flush=True)

    with httpx.Client(
        base_url=args.endpoint.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(10.0, read=600.0),
    ) as client:
        try:
            model = args.model or pick_model(client)
            print(f"Probing {model} at {args.endpoint}", file=sys.stderr)
            fmt = probe(client, model, args.samples, args.max_tokens, on_reply=progress)
        except (httpx.HTTPError, ProbeError) as exc:
            print(f"\nunmimic init: {exc}", file=sys.stderr)
            return 1

    path = fmt.save(settings.data_dir / "models")
    print(f"\nReasoning: {describe(fmt)} ({fmt.matches}/{fmt.samples} replies agree)")
    if fmt.matches < 0.8 * fmt.samples:
        print("Warning: replies disagree on the format; check the config by hand.")
    if fmt.kind == "none":
        print("Warning: no reasoning found; guards will treat each whole reply as the answer.")
    print(f"Wrote {path}")
    return 0


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    import unmimic.guards.input.extraction  # noqa: F401  (registers the plugin)
    import unmimic.guards.output.leak_detector  # noqa: F401  (registers the plugin)
    import unmimic.guards.output.transforms.strip  # noqa: F401  (registers the plugin)
    from unmimic.config import Settings
    from unmimic.plugins import registry
    from unmimic.proxy.app import create_app
    from unmimic.reasoning import load_formats

    settings = Settings()
    upstream = args.upstream or settings.upstream_url
    host = args.host or settings.host
    port = args.port or settings.port
    formats = load_formats(settings.data_dir / "models")

    inputs = []
    for name in settings.input_guards:
        try:
            inputs.append(registry.create("input", name))
        except (ImportError, ValueError) as exc:
            print(f"WARNING: input guard {name} is OFF: {exc}", file=sys.stderr)
    outputs = [registry.create("output", name) for name in settings.output_guards]
    app = create_app(
        upstream,
        input_guards=inputs,
        output_guards=outputs,
        upstream_api_key=settings.upstream_api_key,
        formats=formats,
        upstream_timeout=settings.upstream_timeout,
    )
    print(f"Unmimic proxy on http://{host}:{port}/v1 -> {upstream}", file=sys.stderr)
    print(f"Model formats loaded: {', '.join(formats) or 'none'}", file=sys.stderr)
    print(f"Input guards: {', '.join(g.name for g in inputs) or 'none'}", file=sys.stderr)
    print(f"Output guards: {', '.join(settings.output_guards) or 'none'}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
    return 0


COMMANDS = {"init": init, "serve": serve}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = COMMANDS.get(args.command)
    if command is None:
        print(f"unmimic {args.command}: not implemented yet", file=sys.stderr)
        return NOT_IMPLEMENTED
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
