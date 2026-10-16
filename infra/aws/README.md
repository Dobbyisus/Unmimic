# AWS GPU machine

One g5.xlarge spot machine (NVIDIA A10G, 24 GB) runs the attack generator, the teacher
model and the student training. Everything AWS-side is done with the AWS CLI from
**CloudShell** (the `>_` icon in the console; already signed in, no keys needed).

| Setting  | Value |
|----------|-------|
| Region   | `us-east-1` |
| Machine  | `g5.xlarge` spot, persistent (can be stopped and started), tag `Name=unmimic-gpu` |
| Image    | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), newest |
| Disk     | 150 GB gp3, deleted with the machine |
| Access   | SSH from your IP only (security group `unmimic-ssh`), key pair `unmimic-key` |
| S3       | bucket `unmimic-dobbyisus-2026` via IAM role `unmimic-ec2-role` (no keys on the machine) |
| Safety   | stops itself after 30 idle minutes (no GPU use, nobody logged in), syncing to S3 first |

## Files

| File | Runs on | What it does |
|------|---------|--------------|
| `unmimic-aws.sh` | CloudShell | setup, launch, status, stop, start, allow-ip, terminate |
| `bootstrap.sh` | the machine, first boot (automatic) | auto-shutdown, Python 3.12 + vLLM |
| `push_code.sh` | your laptop (Git Bash) | copies this repo to the machine and installs it |
| `sync_s3.sh` | the machine | copies `data/` results to S3 (or back with `--down`) |

## First time

1. **Laptop, Git Bash**: get your public IP.
   ```bash
   curl -s https://checkip.amazonaws.com
   ```
2. **CloudShell**: Actions > Upload file, twice: `infra/aws/unmimic-aws.sh` and
   `infra/aws/bootstrap.sh`. Then:
   ```bash
   bash unmimic-aws.sh setup <your-ip>
   bash unmimic-aws.sh launch
   ```
   `launch` prints the machine's public IP. First boot installs vLLM (about 10 minutes).
3. **Laptop, Git Bash, in the unmimic folder**: copy the code over and connect.
   ```bash
   bash infra/aws/push_code.sh <public-ip>
   ssh -i ~/.ssh/unmimic-key.pem ubuntu@<public-ip>
   ```
   If `push_code.sh` says first-boot setup is still running, wait and re-run it.
4. **On the machine**: check it worked.
   ```bash
   tail -3 /var/log/unmimic-bootstrap.log
   nvidia-smi
   ```

## Every day

- Start: `bash unmimic-aws.sh start` (CloudShell). The public IP changes on every start.
- Your laptop's IP changed (SSH times out): `bash unmimic-aws.sh allow-ip <new-ip>`.
  On a mobile network that sends SSH out through a different IP than web traffic (Jio
  does), allow the carrier's range instead, e.g. `allow-ip 152.58.0.0/16`. Login still
  needs your key; password login is off on the machine.
- Long job on the CPU only: `touch ~/KEEP_AWAKE` first (GPU jobs keep it awake anyway);
  `rm ~/KEEP_AWAKE` after.
- Done for now: `bash unmimic-aws.sh stop`, or just log out and it stops itself after 30
  idle minutes.
- Project's GPU work finished: `bash unmimic-aws.sh terminate` (deletes the disk; sync to S3
  first with `bash ~/unmimic/infra/aws/sync_s3.sh`).

## Cost

- Running: about $0.40/hour on spot (check the current g5.xlarge spot price).
- Stopped: no GPU charge, but the 150 GB disk costs about $12/month until terminated.
- Spot machines can be reclaimed by AWS with a 2-minute warning. The machine then stops
  (disk kept), so checkpoint long jobs and sync to S3.
- Budget alerts are set at $50 / $150 / $250 (budget `unmimic-spend`).

## Rules

Never point attack prompts at Bedrock or any hosted model API: only at models served on
this machine.
