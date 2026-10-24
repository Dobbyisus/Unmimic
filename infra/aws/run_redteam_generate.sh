#!/bin/bash
# Self red-team, generation step, on the GPU machine and independent of any SSH session:
# serve the generator model with vLLM, run experiments/redteam_generate.py (maintainer-only,
# gitignored; copy it over with push_code.sh), stop vLLM, sync to S3.
# Log: ~/redteam_generate.log. Marker when finished: ~/REDTEAM_GENERATE_DONE.
#
# The generator (Qwen/Qwen3-14B-AWQ, Apache-2.0, ~9 GB at 4 bits) and the teacher do not fit
# on a 24 GB GPU together, so a running teacher is stopped first. Afterwards start the
# teacher again (run_teacher.sh / run_experiment1.sh) to send the new families to it.
#
#   ROUND=2 GEN_ARGS="--hold-out encode_reasoning" nohup setsid bash infra/aws/run_redteam_generate.sh \
#       > ~/redteam_generate.log 2>&1 &
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH

GENERATOR=${GENERATOR:-Qwen/Qwen3-14B-AWQ}
PORT=${PORT:-8001}
GEN_ARGS=${GEN_ARGS:-}
[ -n "${ROUND:-}" ] && GEN_ARGS="--round $ROUND $GEN_ARGS"

( while true; do sleep 900; bash infra/aws/sync_s3.sh >/dev/null 2>&1; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; pkill -f "[v]llm.serve"; bash infra/aws/sync_s3.sh' EXIT

if ! curl -s "localhost:$PORT/v1/models" | grep -q generator; then
  pkill -f "[v]llm.serve" && sleep 10
  echo "=== starting vLLM ($GENERATOR) $(date -u)"
  nohup vllm serve "$GENERATOR" --served-model-name generator \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --port "$PORT" > ~/vllm_generator.log 2>&1 &
  until curl -s "localhost:$PORT/v1/models" | grep -q generator; do
    sleep 20
    pgrep -f "[v]llm.serve" >/dev/null || { echo "vLLM DIED $(date -u)"; exit 1; }
  done
fi

echo "=== generating $(date -u)"
# The input guard's embedder runs on the CPU: vLLM holds the GPU.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="" python experiments/redteam_generate.py \
  --generator-url "http://localhost:$PORT/v1" --model generator $GEN_ARGS || {
  echo "GENERATION FAILED"; exit 1; }

# The generated families are gitignored and live only here until synced.
mkdir -p data/attacks_generated
cp attacks/private/*_v*.yaml data/attacks_generated/ 2>/dev/null
echo "ALL DONE $(date -u)"
touch ~/REDTEAM_GENERATE_DONE
