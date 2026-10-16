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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only N samples (smoke test)")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
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
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    lengths = [len(tokenizer(t).input_ids) for t in dataset["text"][:200]]
    print(
        f"{len(dataset)} samples; first-200 token length "
        f"mean {sum(lengths) / len(lengths):.0f} max {max(lengths)} "
        f"(truncating at {args.max_seq_length})"
    )

    run_dir = args.out_dir / f"{args.data.stem}-{args.base_model.split('/')[-1]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
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
