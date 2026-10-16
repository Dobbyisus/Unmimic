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
- `eval_independent.py` — the input guard on attacks written by real people: Lakera's
  Mosscap and Gandalf password games (MIT license; credit Lakera), never trained on. Scores
  the prompts as written and "retargeted" (the password replaced with "your hidden
  reasoning" by fixed rules). Writes `data/results/input_guard_independent.json`.

- `build_student_data.py` — turns the teacher's saved outputs into one training set per
  condition: `full_reasoning` (what a copier gets with no protection), `unmimic` (the same
  traces passed through the real output guards, so reasoning is stripped and withheld
  responses are dropped) and `answers_only` (the floor). Writes `data/students/`.

GPU machine (`.[experiments]`):

- `teacher_gen.py` — the teacher solves GSM8K / MATH-500 through vLLM; saves question,
  reasoning, answer, correctness and config to `data/traces/`, resumable.
- `train_student.py` — LoRA-fine-tunes a small student (default `Qwen/Qwen2.5-1.5B`,
  Apache-2.0) on one condition's data. Run once per condition with identical settings.
- `eval_student.py` — scores the students on MATH-500 with 95% Clopper-Pearson intervals
  and prints the with/without-Unmimic comparison. Writes `data/results/students.json`.

## The copy-quality experiment, end to end

```bash
# 1. GPU: the teacher solves the problems (slow; resumable)
python experiments/teacher_gen.py --endpoint http://localhost:8000/v1 --dataset gsm8k
python experiments/teacher_gen.py --endpoint http://localhost:8000/v1 --dataset math500

# 2. Anywhere: build each copier's view of that data
python experiments/build_student_data.py --traces data/traces/teacher_gsm8k.jsonl

# 3. GPU: train one student per condition, same settings for each
python experiments/train_student.py --data data/students/teacher_gsm8k_full_reasoning.jsonl
python experiments/train_student.py --data data/students/teacher_gsm8k_unmimic.jsonl

# 4. GPU: benchmark them against each other and the teacher
python experiments/eval_student.py \
    --adapters data/students/models/teacher_gsm8k_full_reasoning-Qwen2.5-1.5B \
               data/students/models/teacher_gsm8k_unmimic-Qwen2.5-1.5B \
    --teacher-traces data/traces/teacher_math500.jsonl
```

Results go to `data/results/` with the config and git hash.
