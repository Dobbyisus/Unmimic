"""Check Unmimic's output side on a real model's saved outputs.

The teacher traces are honest answers to ordinary math problems, so they test the parts
of Unmimic that every normal reply goes through:

1. format detection: would the probe (`unmimic init`) read these replies correctly?
2. strip transform: is the reasoning removed and the answer left intact?
3. leak detector: how often does it withhold an honest answer? Every withhold here is a
   false positive; reported with a 95% Clopper-Pearson upper bound.

python experiments/check_real_outputs.py --traces data/traces/teacher_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from scipy.stats import beta

from unmimic.adapt.probe import describe, observe, vote
from unmimic.guards.output.leak_detector import NOTICE, LeakGuard, leak_score, ngram_overlap
from unmimic.guards.output.transforms.strip import StripReasoning
from unmimic.plugins import Context
from unmimic.reasoning import ModelFormat

TEACHER_FORMAT = ModelFormat(
    kind="tag", open_tag="<think>", close_tag="</think>", implicit_open=True
)


def upper_bound(errors: int, n: int) -> float:
    return 1.0 if errors >= n else float(beta.ppf(0.95, errors + 1, n - errors))


def reply(text: str) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, default=Path("data/traces/teacher_gsm8k.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/results/real_output_checks.json"))
    args = parser.parse_args()

    text = args.traces.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.split("\n") if line.strip()]
    complete = [r for r in rows if r["finish_reason"] != "length" and r["visible_answer"].strip()]
    if args.limit:
        complete = complete[: args.limit]
    print(f"{len(rows)} traces; {len(complete)} complete (reached an answer)")

    # 1. Format detection on every reply, as the probe would see it.
    observations = [observe({"content": r["raw_output"]}) for r in rows]
    kinds = Counter(o.kind for o in observations)
    detected = vote("teacher", observations)
    print(f"\n[format] per-reply: {dict(kinds)}")
    print(
        f"[format] probe would conclude: {describe(detected)} "
        f"({detected.matches}/{detected.samples} agree)"
    )

    # 2. Strip transform: reasoning gone, answer intact.
    strip = StripReasoning()
    strip_bad = []
    for r in complete:
        ctx = Context("check", metadata={"format": TEACHER_FORMAT})
        out = strip.transform(reply(r["raw_output"]), r["reasoning"], ctx)
        content = out["choices"][0]["message"]["content"]
        if "</think>" in content or content.strip() != r["visible_answer"].strip():
            strip_bad.append(r["id"])
    print(
        f"\n[strip] {len(complete) - len(strip_bad)}/{len(complete)} replies reduced to exactly "
        f"the visible answer; mismatches: {strip_bad[:5]}"
    )

    # 3. Leak detector on honest answers: every withhold is a false positive.
    guard = LeakGuard()
    withheld, scores, checks = [], [], Counter()
    started = time.perf_counter()
    for i, r in enumerate(complete, 1):
        ctx = Context("check", metadata={"format": TEACHER_FORMAT})
        out = guard.transform(reply(r["raw_output"]), r["reasoning"], ctx)
        leak = leak_score(r["visible_answer"], r["reasoning"], guard.matcher)
        scores.append(leak.score)
        checks[leak.check] += 1
        if out["choices"][0]["message"]["content"] == NOTICE:
            withheld.append(
                {
                    "id": r["id"],
                    "score": round(leak.score, 3),
                    "check": leak.check,
                    "exact_8gram": round(ngram_overlap(r["visible_answer"], r["reasoning"]), 3),
                    "answer_chars": len(r["visible_answer"]),
                    "answer_start": r["visible_answer"][:160],
                }
            )
        if i % 250 == 0:
            rate = i / (time.perf_counter() - started)
            print(f"  [leak] {i}/{len(complete)} checked, {len(withheld)} withheld ({rate:.1f}/s)")

    n, k = len(complete), len(withheld)
    print(
        f"\n[leak] honest answers withheld: {k}/{n} = {k / n:.2%} "
        f"(95% upper bound {upper_bound(k, n):.2%})"
    )
    print(
        f"[leak] score: median {st.median(scores):.3f}, "
        f"p99 {sorted(scores)[int(0.99 * len(scores))]:.3f}, max {max(scores):.3f}; "
        f"by check: {dict(checks)}"
    )
    answer_lengths = [len(r["visible_answer"]) for r in complete]
    print(
        f"[leak] visible answer length (chars): median {st.median(answer_lengths):.0f}, "
        f"p90 {sorted(answer_lengths)[int(0.9 * len(answer_lengths))]}"
    )
    for w in withheld[:8]:
        print(
            f"  withheld {w['id']}: score {w['score']} via {w['check']} "
            f"(exact {w['exact_8gram']}), answer {w['answer_chars']} chars"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "traces": str(args.traces),
                "n_traces": len(rows),
                "n_complete": n,
                "format_votes": dict(kinds),
                "format_detected": detected.model_dump(),
                "strip_mismatches": strip_bad,
                "leak_withheld": k,
                "leak_false_positive_rate": k / n,
                "leak_false_positive_upper95": upper_bound(k, n),
                "leak_score_median": st.median(scores),
                "withheld": withheld,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
