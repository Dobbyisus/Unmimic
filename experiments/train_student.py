"""Train a copycat student on one condition's data (LoRA; GPU machine).

Every condition is trained with identical settings, so the only difference between the
students is what the copier was able to collect. Run once per condition:

    python experiments/train_student.py --data data/students/teacher_gsm8k_full_reasoning.jsonl
    python experiments/train_student.py --data data/students/teacher_gsm8k_unmimic.jsonl

Writes the adapter to data/students/models/<data stem>-<base model>/ plus a run.json with
the settings and git hash.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B"
# Attention and MLP projections; the same names in Qwen2.5 and Qwen3.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FALLBACK_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{{ eos_token }}{% endfor %}"
)


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="JSONL from build_student_data.py")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=Path("data/students/models"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Teacher replies run to ~7,800 tokens, so examples are long: batch 1 x 16 accumulation
    # keeps the effective batch at 16 while fitting a 24 GB GPU.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="longer examples are dropped, not truncated (truncation cuts off the answer)",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only N samples (smoke test)")
    args = parser.parse_args()

    import math

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        # Base (non-instruct) checkpoints may ship no template.
        tokenizer.chat_template = FALLBACK_TEMPLATE

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    def to_text(row: dict) -> dict:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        # Base checkpoints' templates may not end the reply with EOS; without it the
        # student never learns to stop after the answer.
        if tokenizer.eos_token and not text.rstrip().endswith(tokenizer.eos_token):
            text = text.rstrip() + tokenizer.eos_token
        return {"text": text}

    dataset = dataset.map(
        lambda row: {**to_text(row), "id": row["id"]}, remove_columns=dataset.column_names
    )
    dataset = dataset.map(lambda row: {"tokens": len(tokenizer(row["text"]).input_ids)})
    too_long = [
        i for i, n in zip(dataset["id"], dataset["tokens"], strict=True) if n > args.max_seq_length
    ]
    dataset = dataset.filter(lambda row: row["tokens"] <= args.max_seq_length)
    lengths = sorted(dataset["tokens"])
    print(
        f"{len(dataset)} samples kept, {len(too_long)} dropped as longer than "
        f"{args.max_seq_length} tokens; length median {lengths[len(lengths) // 2]} "
        f"max {lengths[-1]}"
    )
    dataset = dataset.remove_columns(["id", "tokens"])

    run_dir = args.out_dir / f"{args.data.stem}-{args.base_model.split('/')[-1]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # TRL 1.x no longer takes peft_config: attach the LoRA adapter to the model first.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()  # gradient checkpointing with a frozen base model
    model.print_trainable_parameters()

    # transformers 5.x dropped warmup_ratio; 3% of all optimizer steps, as a step count.
    total_steps = math.ceil(len(dataset) / (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = max(1, round(0.03 * total_steps))

    config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        bf16=bf16,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "base_model": args.base_model,
                "samples": len(dataset),
                "dropped_too_long": too_long,
                "epochs": args.epochs,
                "lr": args.lr,
                "effective_batch": args.batch_size * args.grad_accum,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "seed": args.seed,
                "train_loss": result.training_loss,
                "git_hash": git_hash(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"train loss {result.training_loss:.4f}; adapter saved to {run_dir}")


if __name__ == "__main__":
    main()
