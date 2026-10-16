#!/bin/bash
# Copy results, traces, generated attacks and checkpoints from the GPU machine to S3.
# Runs on the machine; uses its IAM role (no keys). Auto-shutdown runs it before stopping.
#   bash ~/unmimic/infra/aws/sync_s3.sh            upload
#   bash ~/unmimic/infra/aws/sync_s3.sh --down     download (e.g. onto a fresh machine)
set -euo pipefail

BUCKET=s3://unmimic-dobbyisus-2026
ROOT=${UNMIMIC_ROOT:-$HOME/unmimic}
DIRS="results traces attacks_generated checkpoints models"

for dir in $DIRS; do
  if [ "${1:-}" = --down ]; then
    aws s3 sync "$BUCKET/data/$dir" "$ROOT/data/$dir" --only-show-errors
  elif [ -d "$ROOT/data/$dir" ]; then
    aws s3 sync "$ROOT/data/$dir" "$BUCKET/data/$dir" --only-show-errors
  fi
done
echo "synced data/{${DIRS// /,}} ${1:+(download)}"
