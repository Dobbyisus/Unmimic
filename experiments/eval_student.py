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

--base-model adds the untrained base model ("Student 0") as a baseline: how much it already
knew before any copying. A base model hasn't learned the \\boxed{} answer format, so
--fewshot N shows it N worked GSM8K training examples first (the usual way base models are
tested on GSM8K), so format errors don't count as math errors:

    python experiments/eval_student.py --base-model Qwen/Qwen2.5-1.5B --fewshot 4 \\
        --max-prompt-tokens 2048 --out data/results/student0_Qwen2.5-1.5B_fewshot.json

Every student gets the same prompt it was trained on and the same token budget. A student
that copies long reasoning needs room to finish, so the budget is large; replies that run
out of room before giving an answer are counted as wrong and reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
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


def fewshot_messages(n: int) -> list[dict]:
    """N worked GSM8K training problems as chat turns, ending in \\boxed{} like the teacher."""
    if not n:
        return []
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, file, *_ = DATASETS["gsm8k"]
    rows = pq.read_table(hf_hub_download(repo, file, repo_type="dataset")).to_pylist()[:n]
    messages = []
    for row in rows:
        steps, answer = row["answer"].split("####")
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()  # GSM8K's calculator annotations
        solution = f"{steps}\nThe final answer is \\boxed{{{answer.strip()}}}."
        messages += [
            {"role": "user", "content": f"{row['question']}\n\n{INSTRUCTION}"},
            {"role": "assistant", "content": solution},
        ]
    return messages


def build_prompt(tokenizer, question: str, shots: list[dict] = ()) -> str:
    messages = [*shots, {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:  # a plain fallback template takes no add_generation_prompt
        return tokenizer.apply_chat_template(messages, tokenize=False) + "assistant: "


def load_run(adapter: Path) -> dict:
    return json.loads((adapter / "run.json").read_text(encoding="utf-8"))


def students(args) -> list[tuple[Path | None, dict]]:
    """(adapter, run info) per model to benchmark; adapter None is the untrained base."""
    models = [(a, {**load_run(a), "name": a.name}) for a in args.adapters]
    if args.base_model:
        name = f"base-{args.base_model.split('/')[-1]}-untrained"
        models.append((None, {"name": name, "base_model": args.base_model, "samples": 0}))
    if not models:
        raise SystemExit("nothing to benchmark: pass --adapters and/or --base-model")
    return models


def score(run: dict, problems: list[dict], outputs: list[tuple[str, bool]]):
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
                "output": text,
            }
        )
    n = len(records)
    correct = sum(r["correct"] for r in records)
    lo, hi = interval(correct, n)
    return {
        "adapter": run["name"],
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


def generate_vllm(models: list[tuple[Path | None, dict]], problems: list[dict], args):
    """All adapters share one loaded base model; each runs as a LoRA on top of it."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    bases = {run["base_model"] for _, run in models}
    if len(bases) != 1:
        raise SystemExit(f"--engine vllm needs one base model for all adapters, got {bases}")
    llm = LLM(
        model=bases.pop(),
        enable_lora=any(adapter for adapter, _ in models),
        max_lora_rank=max((run.get("lora_r", 16) for _, run in models), default=16),
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        stop=STOP,
    )
    shots = fewshot_messages(args.fewshot)
    results = []
    for index, (adapter, run) in enumerate(models, start=1):
        print(f"\n=== {run['name']}  ({run['samples']} training samples)", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter or run["base_model"])
        prompts = [build_prompt(tokenizer, p["question"], shots) for p in problems]
        request = LoRARequest(adapter.name, index, str(adapter)) if adapter else None
        replies = llm.generate(prompts, params, lora_request=request)
        outputs = [(r.outputs[0].text, r.outputs[0].finish_reason == "length") for r in replies]
        result = score(run, problems, outputs)
        print(f"  accuracy {result['accuracy']:.1%}, no answer {result['no_answer_rate']:.1%}")
        results.append(result)
    return results


def generate_hf(models: list[tuple[Path | None, dict]], problems: list[dict], args):
    """Plain transformers fallback: correct but much slower at long lengths."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shots = fewshot_messages(args.fewshot)
    results = []
    for adapter, run in models:
        print(f"\n=== {run['name']}  ({run['samples']} training samples)", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter or run["base_model"])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            run["base_model"],
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP]
        stop_ids = [i for i in stop_ids if isinstance(i, int) and i >= 0] or None

        outputs = []
        for start in range(0, len(problems), args.batch_size):
            batch = problems[start : start + args.batch_size]
            encoded = tokenizer(
                [build_prompt(tokenizer, p["question"], shots) for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids,
                )
            for row in generated:
                new = row[encoded["input_ids"].shape[1] :]
                ended = stop_ids is not None and any(int(t) in stop_ids for t in new)
                text = tokenizer.decode(new, skip_special_tokens=True)
                outputs.append((text, not ended and len(new) >= args.max_new_tokens))
            print(f"  [{len(outputs)}/{len(problems)}]", flush=True)

        result = score(run, problems, outputs)
        print(f"  accuracy {result['accuracy']:.1%}, no answer {result['no_answer_rate']:.1%}")
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapters", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--base-model", default=None, help="also benchmark this base model untrained (Student 0)"
    )
    parser.add_argument(
        "--fewshot", type=int, default=0, help="worked GSM8K training examples in every prompt"
    )
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
    # The teacher's own settings, and DeepSeek's recommendation for R1-style reasoning:
    # greedy decoding (temperature 0) traps reasoning-distilled models in repetition loops
    # (27% of Student A's replies ran to the token limit that way).
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-traces", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/results/students.json"))
    args = parser.parse_args()

    problems = load_problems(args.dataset)
    if args.limit:
        problems = problems[: args.limit]
    print(f"{len(problems)} problems from {args.dataset}; up to {args.max_new_tokens} new tokens")

    generate = generate_vllm if args.engine == "vllm" else generate_hf
    results = generate(students(args), problems, args)

    print(f"\n{'student':48s} {'n':>5s}  accuracy [95% CI]      no answer")
    for r in results:
        lo, hi = r["accuracy_ci95"]
        print(
            f"{r['adapter']:48s} {r['n']:5d}  {r['accuracy']:6.1%} [{lo:5.1%}-{hi:5.1%}]  "
            f"{r['no_answer_rate']:6.1%}"
        )

    teacher = None
    if args.teacher_traces and args.teacher_traces.exists():
        teacher = teacher_reference(args.teacher_traces, problems)
        if teacher:
            lo, hi = teacher["accuracy_ci95"]
            print(
                f"{'teacher (reference)':48s} {teacher['n']:5d}  {teacher['accuracy']:6.1%} "
                f"[{lo:5.1%}-{hi:5.1%}]   (problems 0-{teacher['n'] - 1})"
            )
            ids = set(teacher["ids"])
            print(f"\nSame {teacher['n']} problems as the teacher:")
            for r in results:
                subset = [x for x in r["records"] if x["id"] in ids]
                hits = sum(x["correct"] for x in subset)
                lo, hi = interval(hits, len(subset))
                print(
                    f"{r['adapter']:48s} {len(subset):5d}  {hits / len(subset):6.1%} "
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
    args.out.write_text(
        json.dumps(
            {
                "students": results,
                "teacher_reference": teacher,
                "config": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


def teacher_reference(path: Path, problems: list[dict]) -> dict | None:
    """Teacher accuracy on the longest unbroken run of test problems from the start.

    Traces finish out of order (easy problems first), so a run stopped early holds a
    biased sample beyond that point; only an unbroken prefix 0..n-1 is a fair sample.
    """
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    }
    prefix = []
    for problem in problems:
        if problem["id"] not in rows:
            break
        prefix.append(rows[problem["id"]])
    if not prefix:
        return None
    hits = sum(bool(r.get("correct")) for r in prefix)
    lo, hi = interval(hits, len(prefix))
    return {
        "n": len(prefix),
        "correct": hits,
        "accuracy": hits / len(prefix),
        "accuracy_ci95": [lo, hi],
        "ids": [r["id"] for r in prefix],
    }


if __name__ == "__main__":
    main()
