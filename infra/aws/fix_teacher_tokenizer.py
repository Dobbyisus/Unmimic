"""Build a correct tokenizer for DeepSeek-R1-0528-Qwen3-8B and check it (GPU machine).

The model's tokenizer_config.json names a Llama tokenizer class. transformers 5.x takes that
literally and loads a SentencePiece-style pipeline (Metaspace pre-tokenizer, ByteFallback
decoder) over a byte-level vocabulary. Prompts are then split into the wrong tokens, and
output comes back as raw symbols ("AlbertĠbuys"). This rebuilds the tokenizer straight from
the model's own tokenizer.json, keeps its chat template and special tokens, verifies it, and
saves it for `vllm serve ... --tokenizer <out>`.

python infra/aws/fix_teacher_tokenizer.py            # -> ~/teacher-tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--out", type=Path, default=Path.home() / "teacher-tokenizer")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    file_tokenizer = Tokenizer.from_file(hf_hub_download(args.repo, "tokenizer.json"))
    loaded = AutoTokenizer.from_pretrained(args.repo)
    fixed = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(hf_hub_download(args.repo, "tokenizer.json")),
        bos_token=loaded.bos_token,
        eos_token=loaded.eos_token,
        pad_token=loaded.pad_token,
    )
    fixed.chat_template = loaded.chat_template

    text = "Natalia sold 48 clips.\n\nPlease reason step by step, within \\boxed{}. 2 × 16"
    messages = [{"role": "user", "content": "What is 2+2?"}]
    checks = {
        "token ids match the model's tokenizer.json": fixed.encode(text, add_special_tokens=False)
        == file_tokenizer.encode(text, add_special_tokens=False).ids,
        "decode(encode(text)) == text": fixed.decode(fixed.encode(text, add_special_tokens=False))
        == text,
        "chat template unchanged": fixed.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        == loaded.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    }
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    broken = loaded.encode(text, add_special_tokens=False) != fixed.encode(
        text, add_special_tokens=False
    )
    print(f"default loader {'IS' if broken else 'is not'} mis-tokenizing this model")
    if not all(checks.values()):
        sys.exit(1)
    fixed.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
