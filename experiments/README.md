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

- `teacher_gen.py` — the teacher solves GSM8K problems through vLLM; saves question,
  reasoning, answer, correctness and config to `data/traces/`, resumable. Serve the
  teacher through `infra/aws/run_teacher.sh` (it fixes the teacher's tokenizer).
- `repair_traces.py` — repairs traces saved as byte-level tokenizer symbols ("AlbertĠbuys"),
  produced before the tokenizer fix. Exact; keeps a `.bytelevel.jsonl` backup.
- `check_real_outputs.py` — runs Unmimic's output side over saved teacher replies: format
  detection, strip, and leak-detector false positives on honest answers (95% bound).
- `train_student.py` — LoRA-fine-tunes a small student (default `Qwen/Qwen2.5-1.5B`,
  Apache-2.0) on one condition's data. Run once per condition with identical settings.
- `eval_student.py` — scores the students on GSM8K's held-out test split (1,319 problems,
  none trained on) with 95% Clopper-Pearson intervals and prints the with/without-Unmimic
  comparison. Writes `data/results/students.json`. MATH-500 is not used: about half its
  problems need more reasoning than the ~8k tokens this GPU allows.

## The copy-quality experiment, end to end

```bash
# 1. GPU: the teacher solves the training problems (slow; resumable)
bash infra/aws/run_teacher.sh

# 2. Anywhere: build each copier's view of that data (correct answers only)
python experiments/build_student_data.py --traces data/traces/teacher_gsm8k.jsonl --correct-only

# 3. GPU: train one student per condition, same settings for each
python experiments/train_student.py --data data/students/teacher_gsm8k_full_reasoning.jsonl
python experiments/train_student.py --data data/students/teacher_gsm8k_unmimic.jsonl

# 4. GPU: benchmark them against each other on GSM8K's test split
python experiments/eval_student.py \
    --adapters data/students/models/teacher_gsm8k_full_reasoning-Qwen2.5-1.5B \
               data/students/models/teacher_gsm8k_unmimic-Qwen2.5-1.5B
```

Results go to `data/results/` with the config and git hash.
