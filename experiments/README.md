# Experiments

Data preparation and evaluations. Install with `pip install -e ".[ml]"`; the GPU work
also needs `.[experiments]`.

Runs anywhere (laptop is fine):

- `fetch_wildchat.py` — builds the normal-prompt sets from WildChat-1M (Ai2, ODC-BY
  license; credit Ai2): `data/benign/wildchat_train.jsonl` and the disjoint
  `data/benign/wildchat_eval.jsonl`.
- `eval_input_guard.py` — leave-one-family-out evaluation of the input guard: recall on
  attack families it never saw, false positives on held-back WildChat prompts with a 95%
  upper bound. Writes `data/results/input_guard_heldout.json`.

Planned (GPU machine):

- `teacher_gen.py` — run a self-hosted reasoning model on GSM8K / MATH-500 (and
  MBPP / HumanEval) via vLLM; save prompt, reasoning and answer to JSONL, resumable.
- `train_student.py` — LoRA-fine-tune 0.5B–1.5B students on answers only, full
  reasoning, summarized reasoning, and defended reasoning.
- `eval.py` — score students and the teacher; leak rate vs false-positive rate.

Results go to `data/results/` with the config and git hash.
