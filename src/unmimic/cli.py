"""Command-line interface: ``unmimic init | redteam | calibrate | serve | audit``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

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

    redteam = sub.add_parser("redteam", help="(planned) attack the model and record what leaks")
    redteam.add_argument("--rounds", type=int, default=3)

    calibrate = sub.add_parser(
        "calibrate", help="set guard thresholds for a target false-positive rate"
    )
    calibrate.add_argument(
        "--benign",
        help="normal prompts (JSONL/YAML) for the input guard, NOT ones it trains on "
        "(e.g. data/benign/wildchat_eval.jsonl)",
    )
    calibrate.add_argument(
        "--replies",
        help="JSONL of the model's honest replies (reasoning + answer) for the leak detector",
    )
    calibrate.add_argument("--model", help="model the replies came from, as requests name it")
    calibrate.add_argument(
        "--fpr", type=float, default=0.001, help="target false-positive rate (default 0.001)"
    )
    calibrate.add_argument(
        "--held-out",
        type=float,
        default=0.5,
        help="share of the examples kept back to check the rate (default 0.5)",
    )
    calibrate.add_argument("--seed", type=int, default=0, help="seed for the held-out split")

    serve = sub.add_parser("serve", help="run the protecting proxy")
    serve.add_argument("--upstream", help="upstream model base URL (default: UNMIMIC_UPSTREAM_URL)")
    serve.add_argument("--host", help="interface to listen on (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port to listen on (default: 9000)")

    sub.add_parser("audit", help="(planned) measure how distillable the model's outputs are")

    fetch = sub.add_parser(
        "fetch-benign",
        help="download normal prompts (WildChat-1M, ~450 MB) for the input guard",
    )
    fetch.add_argument("--out", default="data/benign", help="folder to write to")

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
    from unmimic.adapt.calibrate import Calibration
    from unmimic.config import Settings
    from unmimic.plugins import registry
    from unmimic.proxy.app import create_app
    from unmimic.reasoning import load_formats

    settings = Settings()
    upstream = args.upstream or settings.upstream_url
    host = args.host or settings.host
    port = args.port or settings.port
    formats = load_formats(settings.data_dir / "models")
    calibration = Calibration.load(settings.calibration_file)

    inputs = []
    for name in settings.input_guards:
        try:
            inputs.append(registry.create("input", name, **calibration.options(name)))
        except (ImportError, ValueError) as exc:
            print(f"WARNING: input guard {name} is OFF: {exc}", file=sys.stderr)
    outputs = [
        registry.create("output", name, **calibration.options(name))
        for name in settings.output_guards
    ]
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
    if calibration.input_guard:
        threshold = calibration.input_guard.threshold
        print(f"Input guard block threshold: {threshold:.4f}", file=sys.stderr)
    for model, c in calibration.leak_detector.items():
        print(f"Leak threshold for {model}: {c.threshold:.4f}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from unmimic.adapt.calibrate import Calibration
    from unmimic.config import Settings

    if not args.benign and not args.replies:
        print("unmimic calibrate: give --benign and/or --replies", file=sys.stderr)
        return 2
    if args.replies and not args.model:
        print("unmimic calibrate: --replies needs --model", file=sys.stderr)
        return 2

    settings = Settings()
    result = Calibration.load(settings.calibration_file)
    try:
        if args.benign:
            result.input_guard = calibrate_input(Path(args.benign), args, settings)
            report("Input guard (block)", result.input_guard, "blocked")
        if args.replies:
            leak = calibrate_leaks(Path(args.replies), args)
            result.leak_detector[args.model] = leak
            report(f"Leak detector ({args.model})", leak, "withheld")
    except (ImportError, ValueError) as exc:
        print(f"\nunmimic calibrate: {exc}", file=sys.stderr)
        return 1
    result.save(settings.calibration_file)
    print(f"Wrote {settings.calibration_file}; unmimic serve uses it from now on.")
    return 0


def calibrate_input(path: Any, args: argparse.Namespace, settings: Any) -> Any:
    from unmimic.adapt.calibrate import calibrate as run
    from unmimic.corpus import load_prompts
    from unmimic.embeddings import embed_cached
    from unmimic.guards.input.extraction import ExtractionGuard, chunks

    # Prompts the guard trains on are in its own memory (similarity 1.0 to a normal
    # prompt), so they would make it look safer than it is.
    known = set(load_prompts(*settings.benign_files))
    prompts = [p for p in dict.fromkeys(load_prompts(path)) if p not in known]
    print(f"{len(prompts)} normal prompts from {path} (training prompts left out)", file=sys.stderr)
    guard = ExtractionGuard()
    pieces, spans = [], []
    for prompt in prompts:
        parts = chunks(prompt)
        spans.append((len(pieces), len(pieces) + len(parts)))
        pieces += parts
    vectors = embed_cached(pieces, settings.data_dir / "cache")

    def score(span: tuple[int, int]) -> float:
        start, end = span
        return guard.block_score(vectors[start:end]) if end > start else 0.0

    return run(spans, score, args.fpr, args.held_out, args.seed, source=str(path))


def calibrate_leaks(path: Any, args: argparse.Namespace) -> Any:
    from unmimic.adapt.calibrate import calibrate as run
    from unmimic.adapt.calibrate import load_replies
    from unmimic.guards.output.leak_detector import LeakGuard, leak_score

    replies = load_replies(path)
    print(f"{len(replies)} complete replies from {path}", file=sys.stderr)
    guard = LeakGuard()
    if guard.matcher is None:
        raise ImportError("the leak detector's meaning check needs: pip install 'unmimic[ml]'")

    def score(reply: tuple[str, str]) -> float:
        reasoning, answer = reply
        return leak_score(answer, reasoning, guard.matcher).score

    def progress(done: int, total: int) -> None:
        print(f"\r  scored {done}/{total}", end="", file=sys.stderr, flush=True)

    result = run(replies, score, args.fpr, args.held_out, args.seed, str(path), progress)
    print(file=sys.stderr)
    return result


def report(name: str, c: Any, verb: str) -> None:
    print(f"{name}: threshold {c.threshold:.4f} from {c.n_calibration} normal examples")
    if c.held_out:
        h = c.held_out
        print(
            f"  held out: {verb} {h.acted}/{h.n} = {h.rate:.2%} "
            f"(95% upper bound {h.upper95:.2%}; target {c.fpr:.2%})"
        )


def fetch_benign(args: argparse.Namespace) -> int:
    from pathlib import Path

    try:
        from unmimic.benign import fetch
    except ImportError:
        print(
            "unmimic fetch-benign needs the ml extras: pip install 'unmimic[ml]'", file=sys.stderr
        )
        return 1
    try:
        fetch(Path(args.out))
    except (ImportError, ValueError) as exc:
        print(f"unmimic fetch-benign: {exc}", file=sys.stderr)
        return 1
    print("The input guard will use these the next time you run: unmimic serve")
    return 0


COMMANDS = {"init": init, "serve": serve, "calibrate": calibrate, "fetch-benign": fetch_benign}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = COMMANDS.get(args.command)
    if command is None:
        print(f"unmimic {args.command}: not implemented yet", file=sys.stderr)
        return NOT_IMPLEMENTED
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
