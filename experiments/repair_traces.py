"""Repair teacher traces that were saved as byte-level tokenizer symbols.

On the GPU machine, vLLM returned DeepSeek-R1-0528-Qwen3-8B's output without turning the
tokens back into text, so "Albert buys" was saved as "AlbertĠbuys" (Ġ = space, Ċ = newline,
others = bytes of non-ASCII characters). The mapping is GPT-2's byte-to-unicode table, so it
is exactly reversible: each symbol is one byte; the bytes decode as UTF-8.

For every trace this decodes raw_output, then re-derives reasoning, visible_answer,
extracted_answer and correct from the decoded text. The original file is kept once as
<name>.bytelevel.jsonl. Replies that don't decode as UTF-8 are marked decode_error.

python experiments/repair_traces.py data/traces/teacher_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_gen import FORMAT, is_correct, last_boxed  # noqa: E402

from unmimic.reasoning import split_text  # noqa: E402


def byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: printable stand-in character -> byte."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            codes.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, codes, strict=True)}


BYTES = byte_decoder()


def is_byte_level(text: str) -> bool:
    """True when the text uses the stand-in symbols (a real space never appears)."""
    return bool(text) and " " not in text and all(ch in BYTES for ch in text)


def decode(text: str) -> tuple[str, bool]:
    """(decoded text, decode_error). Text that isn't byte-level is returned unchanged."""
    if not is_byte_level(text):
        return text, False
    raw = bytes(BYTES[ch] for ch in text)
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


def repair(row: dict) -> dict:
    text, error = decode(row["raw_output"])
    split = split_text(text, FORMAT)
    extracted = last_boxed(split.answer) or last_boxed(text)
    return {
        **row,
        "raw_output": text,
        "reasoning": split.reasoning,
        "visible_answer": split.answer,
        "extracted_answer": extracted,
        "correct": is_correct(extracted, row["reference"]),
        "decode_error": error,
        "repaired_from_byte_level": text != row["raw_output"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.traces:
        backup = path.with_suffix(".bytelevel.jsonl")
        if not backup.exists():
            shutil.copyfile(path, backup)
        source = backup  # always repair from the untouched original
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        fixed = [repair(r) for r in rows]
        with path.open("w", encoding="utf-8") as f:
            for r in fixed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        changed = sum(r["repaired_from_byte_level"] for r in fixed)
        errors = sum(r["decode_error"] for r in fixed)
        before = sum(bool(r.get("correct")) for r in rows)
        after = sum(r["correct"] for r in fixed)
        print(
            f"{path}: {len(rows)} traces, {changed} decoded, {errors} with decode errors; "
            f"correct before {before}, after {after}; original kept as {backup.name}"
        )


if __name__ == "__main__":
    main()
