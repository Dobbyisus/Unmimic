#!/bin/bash
# Students A vs B on the GPU machine, independent of any SSH session.
# Log: ~/students_run.log. Marker when finished: ~/STUDENTS_RUN_DONE.
#
# 1. frees the GPU (stops a running teacher), 2. smoke test: train + evaluate on 20
# examples and stop if anything fails, 3. train Student A (full reasoning) and Student B
# (through Unmimic) with identical settings, 4. benchmark both on GSM8K's test split with
# the teacher's own score as a reference line. Afterwards the GPU goes idle and
# auto-shutdown stops the machine within ~30 minutes.
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH

DATA=data/students
MODELS=$DATA/models
BASE=${STUDENT_BASE:-Qwen/Qwen2.5-1.5B}
TEACHER_REF=data/traces/teacher_gsm8k_test.jsonl

pkill -f "[v]llm.serve" && sleep 10
( while true; do sleep 900; bash infra/aws/sync_s3.sh >/dev/null 2>&1; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; bash infra/aws/sync_s3.sh' EXIT

echo "=== smoke test started $(date -u)"
python experiments/train_student.py --data $DATA/teacher_gsm8k_full_reasoning.jsonl \
  --base-model "$BASE" --limit 20 --epochs 1 --out-dir $DATA/smoke || { echo "SMOKE TRAIN FAILED"; exit 1; }
python experiments/eval_student.py --adapters $DATA/smoke/teacher_gsm8k_full_reasoning-${BASE##*/} \
  --limit 20 --out data/results/students_smoke.json || { echo "SMOKE EVAL FAILED"; exit 1; }
echo "=== smoke test passed $(date -u)"

for condition in full_reasoning unmimic; do
  if [ -f "$MODELS/teacher_gsm8k_${condition}-${BASE##*/}/run.json" ]; then
    echo "=== $condition already trained, skipping"
    continue
  fi
  echo "=== training $condition started $(date -u)"
  python experiments/train_student.py --data $DATA/teacher_gsm8k_${condition}.jsonl \
    --base-model "$BASE" || { echo "TRAINING $condition FAILED"; exit 1; }
  echo "=== training $condition finished $(date -u)"
done

echo "=== benchmark started $(date -u)"
python experiments/eval_student.py \
  --adapters $MODELS/teacher_gsm8k_full_reasoning-${BASE##*/} $MODELS/teacher_gsm8k_unmimic-${BASE##*/} \
  --teacher-traces $TEACHER_REF --out data/results/students_${BASE##*/}.json || { echo "BENCHMARK FAILED"; exit 1; }
echo "ALL DONE $(date -u)"
touch ~/STUDENTS_RUN_DONE
