# Experiments

Research scripts that run on the GPU machine, not the laptop. Install with
`pip install -e ".[ml,experiments]"`.

Planned:

- `teacher_gen.py` — run a self-hosted reasoning model on GSM8K / MATH-500 (and
  MBPP / HumanEval) via vLLM; save prompt, reasoning and answer to JSONL, resumable.
- `train_student.py` — LoRA-fine-tune 0.5B–1.5B students on answers only, full
  reasoning, summarized reasoning, and defended reasoning.
- `eval.py` — score students and the teacher; leak rate vs false-positive rate.

Results go to `data/results/*.jsonl` with the config and git hash.
