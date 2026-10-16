"""Benchmark the trained students and compare conditions.

Runs each student on a held-out test set (MATH-500 by default) and reports accuracy with a
95% Clopper-Pearson interval, then prints the comparison that is the point of the whole
experiment: a copy trained on the teacher's reasoning vs one trained on what Unmimic lets
through.

    python experiments/eval_student.py \
        --adapters data/students/models/teacher_gsm8k_full_reasoning-Qwen2.5-1.5B \
                   data/students/models/teacher_gsm8k_unmimic-Qwen2.5-1.5B

Add --teacher-traces data/traces/teacher_math500.jsonl to show the teacher's own score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_gen import DATASETS, INSTRUCTION, is_correct, last_boxed, load_problems  # noqa: E402


def interval(hits: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval on a rate."""
    lo = 0.0 if hits == 0 else float(beta.ppf(0.025, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(0.975, hits + 1, n - hits))
    return lo, hi


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:  # a plain fallback template takes no add_generation_prompt
        return tokenizer.apply_chat_template(messages, tokenize=False) + "assistant: "


def evaluate(adapter: Path, problems: list[dict], args) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = json.loads((adapter / "run.json").read_text(encoding="utf-8"))
    base = run["base_model"]
    print(f"\n=== {adapter.name}  (base {base}, {run['samples']} training samples)")

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = PeftModel.from_pretrained(model, adapter).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    records, correct = [], 0
    for start in range(0, len(problems), args.batch_size):
        batch = problems[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, p["question"]) for p in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for problem, output in zip(batch, generated, strict=True):
            text = tokenizer.decode(
                output[encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            predicted = last_boxed(text)
            ok = is_correct(predicted, problem["reference"])
            correct += ok
            records.append(
                {
                    "id": problem["id"],
                    "predicted": predicted,
                    "reference": problem["reference"],
                    "correct": ok,
                    "output_chars": len(text),
                }
            )
        done = len(records)
        print(f"  [{done}/{len(problems)}] accuracy so far {correct / done:.1%}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(records)
    lo, hi = interval(correct, n)
    return {
        "adapter": adapter.name,
        "base_model": base,
        "training_samples": run["samples"],
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_ci95": [lo, hi],
        "mean_output_chars": round(sum(r["output_chars"] for r in records) / n),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapters", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", default="math500", choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--teacher-traces", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/results/students.json"))
    args = parser.parse_args()

    problems = load_problems(args.dataset)
    if args.limit:
        problems = problems[: args.limit]
    print(f"{len(problems)} problems from {args.dataset}")

    results = [evaluate(adapter, problems, args) for adapter in args.adapters]

    print(f"\n{'student':48s} {'n':>5s}  accuracy [95% CI]")
    for r in results:
        lo, hi = r["accuracy_ci95"]
        print(f"{r['adapter']:48s} {r['n']:5d}  {r['accuracy']:6.1%} [{lo:5.1%}-{hi:5.1%}]")

    if args.teacher_traces and args.teacher_traces.exists():
        rows = [
            json.loads(line)
            for line in args.teacher_traces.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        hits = sum(bool(r.get("correct")) for r in rows)
        lo, hi = interval(hits, len(rows))
        print(
            f"{'teacher (reference)':48s} {len(rows):5d}  {hits / len(rows):6.1%} "
            f"[{lo:5.1%}-{hi:5.1%}]"
        )

    by_name = {r["adapter"]: r for r in results}
    full = next((r for n, r in by_name.items() if "full_reasoning" in n), None)
    guarded = next((r for n, r in by_name.items() if "unmimic" in n), None)
    if full and guarded:
        drop = full["accuracy"] - guarded["accuracy"]
        relative = drop / full["accuracy"] if full["accuracy"] else 0.0
        print(
            f"\nCopy trained without Unmimic: {full['accuracy']:.1%}; "
            f"through Unmimic: {guarded['accuracy']:.1%}.\n"
            f"Unmimic cost the copier {drop:.1%} points ({relative:.0%} of its accuracy)."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
