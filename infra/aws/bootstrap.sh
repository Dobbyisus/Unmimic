#!/bin/bash
# First-boot setup for the Unmimic GPU machine. EC2 runs this once as root (user data).
# Log: /var/log/unmimic-bootstrap.log. Finished when /home/ubuntu/BOOTSTRAP_DONE exists.
set -euxo pipefail
exec > /var/log/unmimic-bootstrap.log 2>&1

# 1. Auto-shutdown first, so a forgotten machine never runs for long. Every 5 minutes:
#    busy = GPU in use, someone logged in over SSH, or /home/ubuntu/KEEP_AWAKE exists.
#    After 30 idle minutes: sync results to S3, then shut down (= stop; disk is kept).
cat > /usr/local/bin/unmimic-autoshutdown <<'SHUTDOWN'
#!/bin/bash
IDLE_LIMIT=30
STATE=/var/lib/unmimic-idle-minutes
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
logged_in=$(who | wc -l)
if [ -e /home/ubuntu/KEEP_AWAKE ] || [ "${gpu:-0}" -gt 5 ] || [ "$logged_in" -gt 0 ]; then
  echo 0 > "$STATE"
  exit 0
fi
idle=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 5 ))
echo "$idle" > "$STATE"
if [ "$idle" -ge "$IDLE_LIMIT" ]; then
  logger "unmimic: idle for $idle minutes, syncing to S3 and shutting down"
  sync=/home/ubuntu/unmimic/infra/aws/sync_s3.sh
  [ -x "$sync" ] && UNMIMIC_ROOT=/home/ubuntu/unmimic "$sync" || true
  echo 0 > "$STATE"
  shutdown -h now
fi
SHUTDOWN
chmod +x /usr/local/bin/unmimic-autoshutdown
# The idle count lives on disk; without the @reboot reset, minutes counted before a stop
# carry over and a restarted machine can shut itself down within 5 minutes.
cat > /etc/cron.d/unmimic-autoshutdown <<'CRON'
@reboot root echo 0 > /var/lib/unmimic-idle-minutes
*/5 * * * * root /usr/local/bin/unmimic-autoshutdown
CRON

# 2. Python 3.12 environment with vLLM, for the ubuntu user (uv fetches Python itself).
command -v pip3 >/dev/null || { apt-get update -y && apt-get install -y python3-pip; }
sudo -u ubuntu -H bash -c '
  set -euxo pipefail
  python3 -m pip install --user --upgrade uv || python3 -m pip install --user --break-system-packages uv
  ~/.local/bin/uv venv ~/venv --python 3.12
  # ninja: FlashInfer JIT-compiles sampling kernels at vLLM startup and fails without it.
  ~/.local/bin/uv pip install --python ~/venv/bin/python vllm ninja
  grep -q "source ~/venv/bin/activate" ~/.bashrc || echo "source ~/venv/bin/activate" >> ~/.bashrc
'

# 3. Record what we got, then mark done.
nvidia-smi
sudo -u ubuntu -H /home/ubuntu/venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)" || true
touch /home/ubuntu/BOOTSTRAP_DONE
chown ubuntu:ubuntu /home/ubuntu/BOOTSTRAP_DONE
