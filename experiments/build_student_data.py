"""Build the student training sets from the teacher's saved outputs.

Takes data/traces/teacher_<dataset>.jsonl (from teacher_gen.py) and writes one chat-format
JSONL per condition. The conditions differ only in what the copier managed to collect:

  full_reasoning  what the teacher gives WITHOUT Unmimic: reasoning + answer.
  unmimic         what it gives THROUGH Unmimic: every trace is passed through the real
                  output guards (leak detector, then strip), so the reasoning is removed and
                  withheld responses are dropped, exactly as a copier would see them.
  answers_only    answer only, with no guards. The floor: what any API user always gets.

Comparing students trained on `full_reasoning` and `unmimic` is the headline experiment.

python experiments/build_student_data.py --traces data/traces/teacher_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unmimic.guards.output.leak_detector import NOTICE, LeakGuard
from unmimic.guards.output.transforms.strip import StripReasoning
from unmimic.plugins import Context
from unmimic.reasoning import ModelFormat

# How the teacher marks its reasoning (DeepSeek-R1: the template opens <think> itself).
TEACHER_FORMAT = ModelFormat(
    kind="tag", open_tag="<think>", close_tag="</think>", implicit_open=True
)
CONDITIONS = ("full_reasoning", "unmimic", "answers_only")


def read_traces(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def with_reasoning(trace: dict) -> str:
    """What a copier captures when nothing hides the reasoning."""
    return f"<think>{trace['reasoning'].strip()}</think>\n\n{trace['visible_answer'].strip()}"


def through_unmimic(trace: dict, leak: LeakGuard, strip: StripReasoning) -> tuple[str, dict]:
    """Run one trace through the deployed output guards; returns (text, context metadata).

    Empty text means Unmimic withheld the response, so the copier collects nothing.
    """
    ctx = Context(account_id="copier", model="teacher", metadata={"format": TEACHER_FORMAT})
    response = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": trace["raw_output"]}}]
    }
    response = leak.transform(response, trace["reasoning"], ctx)
    response = strip.transform(response, trace["reasoning"], ctx)
    content = (response["choices"][0]["message"].get("content") or "").strip()
    return ("" if content == NOTICE else content), ctx.metadata


def sample(trace: dict, target: str, condition: str) -> dict:
    return {
        "id": trace["id"],
        "condition": condition,
        "messages": [
            {"role": "user", "content": trace["question"]},
            {"role": "assistant", "content": target},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, default=Path("data/traces/teacher_gsm8k.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/students"))
    parser.add_argument(
        "--conditions", default=",".join(CONDITIONS), help=f"comma-separated: {CONDITIONS}"
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="keep only traces where the teacher got the right answer (a careful copier filters)",
    )
    parser.add_argument(
        "--leak-threshold", type=float, default=None, help="override the leak detector threshold"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="skip the leak detector's multilingual meaning check (faster, exact match only)",
    )
    args = parser.parse_args()

    traces = read_traces(args.traces)
    usable = [
        t
        for t in traces
        if t.get("visible_answer", "").strip()
        and t.get("reasoning", "").strip()
        and (not args.correct_only or t.get("correct"))
    ]
    print(
        f"{len(traces)} traces, {len(usable)} usable"
        + (" (correct only)" if args.correct_only else "")
    )
    if not usable:
        raise SystemExit("no usable traces; check the teacher run finished")

    kwargs = {}
    if args.leak_threshold is not None:
        kwargs["threshold"] = args.leak_threshold
    if args.no_semantic:
        kwargs["semantic_model"] = None
    leak, strip = LeakGuard(**kwargs), StripReasoning()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        if condition not in CONDITIONS:
            raise SystemExit(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
        rows, withheld = [], 0
        for trace in usable:
            if condition == "full_reasoning":
                target = with_reasoning(trace)
            elif condition == "answers_only":
                target = trace["visible_answer"].strip()
            else:
                target, meta = through_unmimic(trace, leak, strip)
                if not target:
                    withheld += 1
                    continue
                if meta.get("leaks"):  # caught but not withheld: should not happen
                    print(f"  note: {trace['id']} flagged a leak yet kept text")
            rows.append(sample(trace, target, condition))

        path = args.out_dir / f"{args.traces.stem}_{condition}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        chars = sum(len(r["messages"][1]["content"]) for r in rows) / max(len(rows), 1)
        summary[condition] = {
            "samples": len(rows),
            "withheld": withheld,
            "mean_target_chars": round(chars),
            "path": str(path),
        }
        print(
            f"{condition:15s} samples={len(rows):5d} withheld={withheld:4d} "
            f"mean target {chars:6.0f} chars -> {path}"
        )

    out = args.out_dir / f"{args.traces.stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
