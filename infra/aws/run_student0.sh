#!/bin/bash
# Student 0 on the GPU machine: untrained base models on GSM8K's test split, same budget
# and sampling as Students A and B. For each base model (default Qwen2.5-1.5B, the
# students' base, then Qwen2.5-0.5B, a candidate weaker student):
#   few-shot  — 4 worked examples first, all 1,319 problems. The standard way to test base
#               models on GSM8K: it shows the \boxed{} format and where to stop.
#   zero-shot — the students' exact prompt, first 200 problems only. Untrained models
#               don't stop after answering (they write on into invented questions, and the
#               grader reads the last \boxed{}), so this run is mainly for inspecting outputs.
# Log: ~/student0_run.log. Marker when finished: ~/STUDENT0_RUN_DONE.
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH

BASES=${STUDENT0_BASES:-"Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-0.5B"}
TEACHER_REF=data/traces/teacher_gsm8k_test.jsonl

pkill -f "[v]llm.serve" && sleep 10
trap 'bash infra/aws/sync_s3.sh' EXIT

for base in $BASES; do
  name=${base##*/}
  echo "=== $name few-shot started $(date -u)"
  python experiments/eval_student.py --base-model "$base" --fewshot 4 --max-prompt-tokens 2048 \
    --teacher-traces $TEACHER_REF --out data/results/student0_${name}_fewshot.json \
    || { echo "$name FEW-SHOT FAILED"; exit 1; }
  echo "=== $name zero-shot (200) started $(date -u)"
  python experiments/eval_student.py --base-model "$base" --limit 200 \
    --out data/results/student0_${name}_zeroshot200.json || { echo "$name ZERO-SHOT FAILED"; exit 1; }
done
echo "ALL DONE $(date -u)"
touch ~/STUDENT0_RUN_DONE
