"""Benchmark the trained students and compare conditions.

Runs each student on GSM8K's held-out test split (1,319 problems, none of them in the
training traces) and reports accuracy with a 95% Clopper-Pearson interval, then prints the
comparison that is the point of the whole experiment: a copy trained on the teacher's
reasoning vs one trained on what Unmimic lets through.

    python experiments/eval_student.py \
        --adapters data/students/models/teacher_gsm8k_full_reasoning-Qwen2.5-1.5B \
                   data/students/models/teacher_gsm8k_unmimic-Qwen2.5-1.5B

MATH-500 was dropped as a benchmark: about half its problems need more than the ~8k tokens
of reasoning this GPU allows, so it would penalise the student that learned to reason.
--teacher-traces adds a reference line from saved teacher traces of the same test set.

Every student gets the same prompt it was trained on and the same token budget. A student
that copies long reasoning needs room to finish, so the budget is large; replies that run
out of room before giving an answer are counted as wrong and reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_gen import DATASETS, INSTRUCTION, is_correct, last_boxed, load_problems  # noqa: E402

# Qwen chat turns end with <|im_end|>; base checkpoints don't treat it as a stop token.
STOP = ["<|im_end|>", "<|endoftext|>"]


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


def load_run(adapter: Path) -> dict:
    return json.loads((adapter / "run.json").read_text(encoding="utf-8"))


def score(adapter: Path, run: dict, problems: list[dict], outputs: list[tuple[str, bool]]):
    """outputs: (generated text, ran out of tokens) per problem, in order."""
    records = []
    for problem, (text, cut_off) in zip(problems, outputs, strict=True):
        predicted = last_boxed(text)
        records.append(
            {
                "id": problem["id"],
                "predicted": predicted,
                "reference": problem["reference"],
                "correct": is_correct(predicted, problem["reference"]),
                "no_answer": predicted is None,
                "cut_off": cut_off,
                "output_chars": len(text),
            }
        )
    n = len(records)
    correct = sum(r["correct"] for r in records)
    lo, hi = interval(correct, n)
    return {
        "adapter": adapter.name,
        "base_model": run["base_model"],
        "training_samples": run["samples"],
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_ci95": [lo, hi],
        "no_answer_rate": sum(r["no_answer"] for r in records) / n,
        "cut_off_rate": sum(r["cut_off"] for r in records) / n,
        "mean_output_chars": round(sum(r["output_chars"] for r in records) / n),
        "records": records,
    }


def generate_vllm(adapters: list[Path], problems: list[dict], args) -> list[dict]:
    """All adapters share one loaded base model; each runs as a LoRA on top of it."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    runs = [load_run(a) for a in adapters]
    bases = {r["base_model"] for r in runs}
    if len(bases) != 1:
        raise SystemExit(f"--engine vllm needs one base model for all adapters, got {bases}")
    llm = LLM(
        model=bases.pop(),
        enable_lora=True,
        max_lora_rank=max(r.get("lora_r", 16) for r in runs),
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop=STOP)
    results = []
    for index, (adapter, run) in enumerate(zip(adapters, runs, strict=True), start=1):
        print(f"\n=== {adapter.name}  ({run['samples']} training samples)", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter)
        prompts = [build_prompt(tokenizer, p["question"]) for p in problems]
        replies = llm.generate(
            prompts, params, lora_request=LoRARequest(adapter.name, index, str(adapter))
        )
        outputs = [(r.outputs[0].text, r.outputs[0].finish_reason == "length") for r in replies]
        result = score(adapter, run, problems, outputs)
        print(f"  accuracy {result['accuracy']:.1%}, no answer {result['no_answer_rate']:.1%}")
        results.append(result)
    return results


def generate_hf(adapters: list[Path], problems: list[dict], args) -> list[dict]:
    """Plain transformers fallback: correct but much slower at long lengths."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = []
    for adapter in adapters:
        run = load_run(adapter)
        print(f"\n=== {adapter.name}  ({run['samples']} training samples)", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            run["base_model"],
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = PeftModel.from_pretrained(model, adapter).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP]
        stop_ids = [i for i in stop_ids if isinstance(i, int) and i >= 0] or None

        outputs = []
        for start in range(0, len(problems), args.batch_size):
            batch = problems[start : start + args.batch_size]
            encoded = tokenizer(
                [build_prompt(tokenizer, p["question"]) for p in batch],
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
                    eos_token_id=stop_ids,
                )
            for row in generated:
                new = row[encoded["input_ids"].shape[1] :]
                ended = stop_ids is not None and any(int(t) in stop_ids for t in new)
                text = tokenizer.decode(new, skip_special_tokens=True)
                outputs.append((text, not ended and len(new) >= args.max_new_tokens))
            print(f"  [{len(outputs)}/{len(problems)}]", flush=True)

        result = score(adapter, run, problems, outputs)
        print(f"  accuracy {result['accuracy']:.1%}, no answer {result['no_answer_rate']:.1%}")
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapters", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", default="gsm8k_test", choices=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=16, help="hf engine only")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=7680,
        help="same for every student; long enough to finish copied reasoning",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--teacher-traces", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/results/students.json"))
    args = parser.parse_args()

    problems = load_problems(args.dataset)
    if args.limit:
        problems = problems[: args.limit]
    print(f"{len(problems)} problems from {args.dataset}; up to {args.max_new_tokens} new tokens")

    generate = generate_vllm if args.engine == "vllm" else generate_hf
    results = generate(args.adapters, problems, args)

    print(f"\n{'student':48s} {'n':>5s}  accuracy [95% CI]      no answer")
    for r in results:
        lo, hi = r["accuracy_ci95"]
        print(
            f"{r['adapter']:48s} {r['n']:5d}  {r['accuracy']:6.1%} [{lo:5.1%}-{hi:5.1%}]  "
            f"{r['no_answer_rate']:6.1%}"
        )

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
