#!/bin/bash
# Full teacher run on the GPU machine, independent of any SSH session.
# Log: ~/teacher_run.log. Marker when finished: ~/TEACHER_RUN_DONE.
# Afterwards the GPU goes idle and auto-shutdown stops the machine within ~30 minutes.
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH

# The pilot used a 4096-token cap; keep it apart so the full run is consistent.
if [ -f data/traces/teacher_gsm8k.jsonl ] && [ ! -f data/traces/teacher_gsm8k_pilot.jsonl ]; then
  mv data/traces/teacher_gsm8k.jsonl data/traces/teacher_gsm8k_pilot.jsonl
fi

# Serve the teacher if it isn't already (e.g. after a reboot). transformers 5.x loads this
# model's tokenizer wrongly (prompts mis-split, output as "AlbertĠbuys"), so vLLM gets a
# corrected copy; the script refuses to continue if the corrected copy fails its checks.
if ! curl -s localhost:8000/v1/models | grep -q teacher; then
  python infra/aws/fix_teacher_tokenizer.py --out ~/teacher-tokenizer || exit 1
  echo "starting vLLM $(date -u)"
  nohup vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --served-model-name teacher \
    --tokenizer ~/teacher-tokenizer \
    --max-model-len 8192 --gpu-memory-utilization 0.92 --port 8000 > ~/vllm_teacher.log 2>&1 &
  until curl -s localhost:8000/v1/models | grep -q teacher; do
    sleep 20
    pgrep -f "[v]llm.serve" >/dev/null || { echo "vLLM died $(date -u)"; exit 1; }
  done
fi

# Copy results to S3 every 15 minutes while running.
( while true; do sleep 900; bash infra/aws/sync_s3.sh >/dev/null 2>&1; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; bash infra/aws/sync_s3.sh' EXIT

# TEACHER_DATASETS: which teacher_gen datasets to run, in order (default: the training
# traces). E.g. TEACHER_DATASETS=gsm8k_test for the teacher's reference score.
# GSM8K_LIMIT: how many GSM8K training problems to solve (default 3000 of 7473). The run is
# resumable, so raising it later only solves the extra problems.
TEACHER_DATASETS=${TEACHER_DATASETS:-gsm8k}
GSM8K_LIMIT=${GSM8K_LIMIT:-3000}
for dataset in $TEACHER_DATASETS; do
  limit=()
  [ "$dataset" = gsm8k ] && limit=(--limit "$GSM8K_LIMIT")
  echo "=== $dataset started $(date -u) ${limit[*]}"
  python experiments/teacher_gen.py --endpoint http://localhost:8000/v1 --model teacher \
    --dataset "$dataset" --max-tokens 7800 --concurrency 32 --timeout 3600 "${limit[@]}"
  echo "=== $dataset finished $(date -u)"
done
echo "ALL DONE $(date -u)"
touch ~/TEACHER_RUN_DONE
