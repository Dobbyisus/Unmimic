"""Teacher outputs: the teacher model solves math problems; every output is saved in full.

Sends GSM8K (train, MIT license) and MATH-500 (test) problems to an OpenAI-compatible
server (vLLM on the GPU machine, see run_teacher_gen.sh) and writes one JSON line per
problem to data/traces/teacher_<dataset>.jsonl:

    question, reference answer, raw output, reasoning, visible answer, finish reason,
    token count, whether the answer is correct, and the config + git hash.

Resumable: problems already in the output file are skipped. The raw output is kept so the
student experiment can later pass the same outputs through Unmimic.

python experiments/teacher_gen.py --endpoint http://localhost:8000/v1 --dataset gsm8k --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import time
from pathlib import Path

import httpx

from unmimic.reasoning import ModelFormat, split_text

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# DeepSeek-R1 models: the chat template may open <think> itself, so the output can start
# mid-reasoning and only contain </think>.
FORMAT = ModelFormat(kind="tag", open_tag="<think>", close_tag="</think>", implicit_open=True)
DATASETS = {
    # name: (repo, file, question field, answer field, max new tokens)
    "gsm8k": ("openai/gsm8k", "main/train-00000-of-00001.parquet", "question", "answer", 4096),
    "math500": ("HuggingFaceH4/MATH-500", "test.jsonl", "problem", "answer", 7168),
}


def load_problems(name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    repo, file, q_field, a_field, _ = DATASETS[name]
    path = hf_hub_download(repo, file, repo_type="dataset")
    if file.endswith(".parquet"):
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    else:
        rows = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line
        ]
    problems = []
    for i, row in enumerate(rows):
        reference = row[a_field]
        if name == "gsm8k":
            reference = reference.split("####")[-1].strip()
        problems.append({"id": f"{name}-{i}", "question": row[q_field], "reference": reference})
    return problems


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def normalize_answer(ans: str | None) -> str:
    """Normalize math/numeric answers for comparison across GSM8K and MATH-500."""
    if ans is None:
        return ""
    s = ans.strip()
    # Strip surrounding math delimiters and units
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Remove common LaTeX formatting wrappers
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\$", "").replace("\\%", "%")
    # Normalize \dfrac and \tfrac to \frac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Remove thousands commas inside numbers (e.g., 1,234 -> 1234)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # Collapse all whitespace
    s = re.sub(r"\s+", "", s)
    # Attempt numeric normalization (e.g. 42.0 == 42)
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.10g}"
    except ValueError:
        return s


def is_correct(predicted: str | None, reference: str) -> bool:
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(reference)


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


async def detect_model(client: httpx.AsyncClient, endpoint: str) -> str:
    resp = await client.get(f"{endpoint}/models")
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models reported by {endpoint}/models")
    return data[0]["id"]


async def solve_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    problem: dict,
    max_tokens: int,
    temperature: float,
    top_p: float,
    config_meta: dict,
) -> dict:
    prompt = f"{problem['question']}\n\n{INSTRUCTION}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    t0 = time.monotonic()
    resp = await client.post(f"{endpoint}/chat/completions", json=payload)
    resp.raise_for_status()
    elapsed = round(time.monotonic() - t0, 3)

    body = resp.json()
    choice = body["choices"][0]
    message = choice.get("message", {})

    # Handle servers that either keep <think> in content or split reasoning_content
    raw_output = message.get("content") or ""
    reasoning_field = message.get("reasoning_content") or message.get("reasoning")
    if reasoning_field and "</think>" not in raw_output:
        raw_output = f"<think>{reasoning_field}</think>{raw_output}"

    split = split_text(raw_output, FORMAT)
    # Support both tuple unpacking and object attribute access from unmimic.reasoning.split_text
    if isinstance(split, tuple):
        reasoning, visible = split[0], split[1]
    else:
        reasoning = getattr(split, "reasoning", "")
        visible = split.answer

    extracted = last_boxed(visible) or last_boxed(raw_output)
    correct = is_correct(extracted, problem["reference"])

    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens")

    return {
        "id": problem["id"],
        "question": problem["question"],
        "reference": problem["reference"],
        "raw_output": raw_output,
        "reasoning": reasoning,
        "visible_answer": visible,
        "extracted_answer": extracted,
        "correct": correct,
        "finish_reason": choice.get("finish_reason"),
        "token_count": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "elapsed_s": elapsed,
        "config": config_meta,
    }


async def run(args: argparse.Namespace) -> None:
    endpoint = args.endpoint.rstrip("/")
    _, _, _, _, default_max_tokens = DATASETS[args.dataset]
    max_tokens = args.max_tokens or default_max_tokens

    out_path = Path(args.output or f"data/traces/teacher_{args.dataset}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already completed problem IDs for resumption
    completed_ids: set[str] = set()
    correct_so_far = 0
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            completed_ids.add(record["id"])
            if record.get("correct"):
                correct_so_far += 1

    problems = load_problems(args.dataset)
    if args.limit is not None:
        problems = problems[: args.limit]

    pending = [p for p in problems if p["id"] not in completed_ids]
    print(
        f"Dataset: {args.dataset} | Total target: {len(problems)} | "
        f"Already done: {len(completed_ids)} | Pending: {len(pending)}"
    )
    if not pending:
        print("All problems already completed.")
        return

    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        model = args.model or await detect_model(client, endpoint)
        git_hash = get_git_hash()

        config_meta = {
            "model": model,
            "dataset": args.dataset,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "git_hash": git_hash,
        }
        print(f"Using model: {model} (git: {git_hash})")

        sem = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        done_count = len(completed_ids)
        total_count = len(problems)

        with out_path.open("a", encoding="utf-8") as f:

            async def worker(prob: dict) -> None:
                nonlocal done_count, correct_so_far
                async with sem:
                    for attempt in range(1, args.retries + 1):
                        try:
                            record = await solve_one(
                                client=client,
                                endpoint=endpoint,
                                model=model,
                                problem=prob,
                                max_tokens=max_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                config_meta=config_meta,
                            )
                            break
                        except Exception as exc:
                            if attempt == args.retries:
                                print(
                                    f"[ERROR] {prob['id']} failed after "
                                    f"{args.retries} attempts: {exc}"
                                )
                                return
                            await asyncio.sleep(2 ** (attempt - 1))

                    async with write_lock:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        done_count += 1
                        if record["correct"]:
                            correct_so_far += 1
                        acc = (correct_so_far / done_count) * 100
                        print(
                            f"[{done_count}/{total_count}] {prob['id']} | "
                            f"pred={record['extracted_answer']!r} ref={record['reference']!r} | "
                            f"ok={record['correct']} | tokens={record['token_count']} | "
                            f"acc={acc:.1f}%"
                        )

            await asyncio.gather(*(worker(p) for p in pending))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (auto-detected from /v1/models if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/traces/teacher_<dataset>.jsonl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of problems to evaluate"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Override default dataset max new tokens"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Number of concurrent requests to vLLM"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds")
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retries per failed problem"
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
