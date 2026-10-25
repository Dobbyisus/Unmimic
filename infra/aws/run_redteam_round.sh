#!/bin/bash
# One self red-team round on the GPU machine, independent of any SSH session:
#   1. generate: serve the generator (Qwen/Qwen3-14B-AWQ) and run experiments/redteam_generate.py
#      (maintainer-only, gitignored), which writes attacks/private/<type>_v<ROUND>.yaml;
#   2. sample: up to SAMPLE_PER_TYPE new attacks per type into data/redteam/round_<ROUND>/send/
#      (only this round's attacks are sent; the old corpus was measured by Experiment 1);
#   3. teacher: serve the teacher (fixed tokenizer) and the proxy with the input guard OFF (it
#      trains on attacks/private at start, so it would block the new attacks by memorization;
#      the input side is measured by the generator's screening verdicts instead), then send each
#      sampled attack to the teacher directly and through Unmimic (run_experiment1.py);
#   4. score: score_experiment1.py -> scores, an audit file, and one label per reply
#      (experiment1_labels.jsonl, which the next round reads with --successes).
# Each stage is skipped when its output exists, so rerunning continues where it stopped.
# The two models don't fit on a 24 GB GPU together, so they run one after the other.
#
#   ROUND=2 SAMPLE_PER_TYPE=20 GEN_ARGS="--hold-out encode_reasoning" \
#     nohup setsid bash infra/aws/run_redteam_round.sh > ~/redteam_round.log 2>&1 &
# Marker when finished: ~/REDTEAM_ROUND_DONE.
set -u
cd ~/unmimic
source ~/venv/bin/activate
export PATH=$HOME/venv/bin:$PATH

GENERATOR=${GENERATOR:-Qwen/Qwen3-14B-AWQ}
GEN_PORT=${GEN_PORT:-8001}
GEN_CONCURRENCY=${GEN_CONCURRENCY:-32}
GEN_ARGS=${GEN_ARGS:-}
SAMPLE_PER_TYPE=${SAMPLE_PER_TYPE:-20}
CONCURRENCY=${CONCURRENCY:-16}
# 8-bit KV cache fits 2-3x more replies at once on an L4 (g6). On an A10G (g5) use KV_DTYPE=auto
# (16-bit) if vLLM rejects fp8 there.
KV_DTYPE=${KV_DTYPE:-fp8}
ROUND=${ROUND:-$(python experiments/redteam_generate.py --print-next-round 2>/dev/null)}
[ -n "$ROUND" ] || { echo "could not work out the round number"; exit 1; }
RD=data/redteam/round_$ROUND
echo "=== red-team round $ROUND $(date -u)"
rm -f ~/REDTEAM_ROUND_DONE

( while true; do sleep 900; bash infra/aws/sync_s3.sh >/dev/null 2>&1; done ) &
SYNC=$!
trap 'kill $SYNC 2>/dev/null; pkill -f "[u]nmimic serve"; pkill -f "[v]llm.serve"; bash infra/aws/sync_s3.sh' EXIT

wait_for() {  # wait_for <port> <served name> <log>
  until curl -s "localhost:$1/v1/models" | grep -q "$2"; do
    sleep 20
    pgrep -f "[v]llm.serve" >/dev/null || { echo "vLLM DIED $(date -u); see $3"; tail -20 "$3"; exit 1; }
  done
}

# --- 1. generate ------------------------------------------------------------------------
if [ ! -f "$RD/GENERATED" ]; then
  pkill -f "[v]llm.serve" && sleep 15
  echo "=== starting generator ($GENERATOR) $(date -u)"
  nohup vllm serve "$GENERATOR" --served-model-name generator \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --port "$GEN_PORT" > ~/vllm_generator.log 2>&1 &
  wait_for "$GEN_PORT" generator ~/vllm_generator.log
  echo "=== generating $(date -u)"
  # The input guard's embedder (screening) runs on the CPU: vLLM holds the GPU.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="" python experiments/redteam_generate.py --round "$ROUND" \
    --generator-url "http://localhost:$GEN_PORT/v1" --model generator \
    --concurrency "$GEN_CONCURRENCY" $GEN_ARGS || { echo "GENERATION FAILED"; exit 1; }
  touch "$RD/GENERATED"
  pkill -f "[v]llm.serve" && sleep 15
fi
mkdir -p data/attacks_generated
cp attacks/private/*_v"$ROUND".yaml data/attacks_generated/ 2>/dev/null

# --- 2. sample --------------------------------------------------------------------------
if [ ! -d "$RD/send" ]; then
  python - "$ROUND" "$SAMPLE_PER_TYPE" "$RD/send" <<'EOF' || { echo "SAMPLING FAILED"; exit 1; }
import random, sys
from pathlib import Path
import yaml
round_number, per_type, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
files = sorted(Path("attacks/private").glob(f"*_v{round_number}.yaml"))
if not files:
    raise SystemExit(f"no round {round_number} families in attacks/private")
out.mkdir(parents=True)
rng = random.Random(0)
for path in files:
    family = yaml.safe_load(path.read_text(encoding="utf-8"))
    examples = list(family["examples"])
    picked = examples if len(examples) <= per_type else rng.sample(examples, per_type)
    family["examples"] = picked
    (out / path.name).write_text(
        yaml.safe_dump(family, sort_keys=False, allow_unicode=True, width=1000), encoding="utf-8"
    )
    print(f"  {family['family']}: sending {len(picked)} of {len(examples)}")
EOF
fi

# --- 3. teacher -------------------------------------------------------------------------
if [ ! -f "$RD/TEACHER_DONE" ]; then
  if ! curl -s localhost:8000/v1/models | grep -q teacher; then
    pkill -f "[v]llm.serve" && sleep 15
    python infra/aws/fix_teacher_tokenizer.py --out ~/teacher-tokenizer || exit 1
    echo "=== starting teacher $(date -u)"
    nohup vllm serve deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --served-model-name teacher \
      --tokenizer ~/teacher-tokenizer --kv-cache-dtype "$KV_DTYPE" \
      --max-model-len 8192 --gpu-memory-utilization 0.93 --port 8000 > ~/vllm_teacher.log 2>&1 &
    wait_for 8000 teacher ~/vllm_teacher.log
  fi
  if [ ! -f data/models/teacher.json ]; then
    unmimic init --endpoint http://localhost:8000/v1 --model teacher --samples 5 || {
      echo "PROBE FAILED"; exit 1; }
  fi
  if ! curl -s localhost:9000/v1/models | grep -q teacher; then
    echo "=== starting proxy (input guard off) $(date -u)"
    CUDA_VISIBLE_DEVICES="" UNMIMIC_INPUT_GUARDS='[]' nohup unmimic serve \
      --upstream http://localhost:8000/v1 > ~/unmimic_proxy.log 2>&1 &
    for _ in $(seq 1 60); do
      curl -s localhost:9000/v1/models | grep -q teacher && break
      sleep 5
    done
    curl -s localhost:9000/v1/models | grep -q teacher || { echo "PROXY FAILED"; exit 1; }
  fi
  echo "=== sending round $ROUND attacks to the teacher $(date -u)"
  python experiments/run_experiment1.py --model teacher --concurrency "$CONCURRENCY" \
    --attack-dirs "$RD/send" --output-dir "$RD/teacher" || { echo "TEACHER RUN FAILED"; exit 1; }
  touch "$RD/TEACHER_DONE"
fi
pkill -f "[u]nmimic serve"
pkill -f "[v]llm.serve" && sleep 15

# --- 4. score ---------------------------------------------------------------------------
echo "=== scoring $(date -u)"
python experiments/score_experiment1.py --results-dir "$RD/teacher" || { echo "SCORING FAILED"; exit 1; }
echo "ALL DONE $(date -u)"
touch ~/REDTEAM_ROUND_DONE
