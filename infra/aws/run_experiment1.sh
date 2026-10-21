#!/bin/bash
# Experiment 1 on the GPU machine, independent of any SSH session: every attack in
# attacks/public + attacks/private sent to the teacher directly and through Unmimic, then
# scored. Log: ~/experiment1_run.log. Marker when finished: ~/EXPERIMENT1_RUN_DONE.
#
# 1. serve the teacher (fixed tokenizer) on :8000, leaving GPU memory for the leak
#    detector's embedding model; 2. `unmimic init` so the proxy knows the <think> format
#    (without it the proxy sees no reasoning and protects nothing); 3. the proxy on :9000
#    with the input guard OFF (it trained on exactly these prompts and would block them all
#    by memorization), so only the output guards are measured; 4. run_experiment1.py;
#    5. stop vLLM and score with score_experiment1.py.
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH
CONCURRENCY=${CONCURRENCY:-16}

( while true; do sleep 900; bash infra/aws/sync_s3.sh >/dev/null 2>&1; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; pkill -f "[u]nmimic serve"; bash infra/aws/sync_s3.sh' EXIT

if ! curl -s localhost:8000/v1/models | grep -q teacher; then
  python infra/aws/fix_teacher_tokenizer.py --out ~/teacher-tokenizer || exit 1
  echo "=== starting vLLM $(date -u)"
  # On a 24 GB GPU the 16 GB model leaves room for only ~11k tokens of KV cache at 16 bits
  # (1.4 replies of 8k tokens at once: ~70 tok/s on an L4). An 8-bit KV cache and a bigger
  # memory share fit 2-3x more replies at once.
  nohup vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --served-model-name teacher \
    --tokenizer ~/teacher-tokenizer --kv-cache-dtype fp8 \
    --max-model-len 8192 --gpu-memory-utilization 0.93 --port 8000 > ~/vllm_teacher.log 2>&1 &
  until curl -s localhost:8000/v1/models | grep -q teacher; do
    sleep 20
    pgrep -f "[v]llm.serve" >/dev/null || { echo "vLLM DIED $(date -u)"; exit 1; }
  done
fi

if [ ! -f data/models/teacher.json ]; then
  echo "=== format probe $(date -u)"
  unmimic init --endpoint http://localhost:8000/v1 --model teacher --samples 5 || {
    echo "PROBE FAILED"; exit 1; }
fi
cat data/models/teacher.json

echo "=== starting proxy $(date -u)"
# The proxy's embedding model runs on the CPU: vLLM holds nearly all GPU memory.
CUDA_VISIBLE_DEVICES="" UNMIMIC_INPUT_GUARDS='[]' nohup unmimic serve \
  --upstream http://localhost:8000/v1 > ~/unmimic_proxy.log 2>&1 &
for _ in $(seq 1 60); do
  curl -s localhost:9000/v1/models | grep -q teacher && break
  sleep 5
done
curl -s localhost:9000/v1/models | grep -q teacher || { echo "PROXY FAILED"; exit 1; }

echo "=== attacks started $(date -u)"
python experiments/run_experiment1.py --model teacher --concurrency "$CONCURRENCY" || {
  echo "ATTACK RUN FAILED"; exit 1; }

echo "=== scoring started $(date -u)"
pkill -f "[u]nmimic serve"
pkill -f "[v]llm.serve" && sleep 15
python experiments/score_experiment1.py || { echo "SCORING FAILED"; exit 1; }
echo "ALL DONE $(date -u)"
touch ~/EXPERIMENT1_RUN_DONE
