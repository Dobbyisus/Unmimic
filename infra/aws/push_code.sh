#!/usr/bin/env bash
# Copy this repo to the GPU machine and install it there. Run on your laptop, in Git Bash,
# from the unmimic folder:   bash infra/aws/push_code.sh <public-ip>
#
# Skips .venv, data/ and caches. Safe to re-run after every code change.
set -euo pipefail

IP=${1:?usage: bash infra/aws/push_code.sh <public-ip>  (bash unmimic-aws.sh status shows it)}
KEY=${UNMIMIC_KEY:-$HOME/.ssh/unmimic-key.pem}
[ -f "$KEY" ] || { echo "key not found: $KEY (set UNMIMIC_KEY to its path)" >&2; exit 1; }
cd "$(dirname "$0")/../.."

tar czf - --exclude=./.venv --exclude=./data --exclude='__pycache__' \
    --exclude=./.ruff_cache --exclude=./.pytest_cache . |
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$IP" '
    set -e
    test -e ~/BOOTSTRAP_DONE || { echo "first-boot setup still running; try again in a few minutes (log: /var/log/unmimic-bootstrap.log)"; exit 1; }
    mkdir -p ~/unmimic && tar xzf - -C ~/unmimic
    chmod +x ~/unmimic/infra/aws/*.sh
    ~/.local/bin/uv pip install --python ~/venv/bin/python -q -e "$HOME/unmimic[ml,experiments]"
    echo "code in ~/unmimic, installed into ~/venv"
  '
