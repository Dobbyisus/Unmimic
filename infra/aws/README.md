# AWS

GPU machine for teacher traces, student training and red-team runs.

| Setting  | Value                                  |
|----------|----------------------------------------|
| Region   | `us-east-1`                            |
| Instance | `g5.xlarge` (A10G 24 GB), spot preferred |
| AMI      | Deep Learning AMI (PyTorch, Ubuntu)    |
| Bucket   | `unmimic-dobbyisus-2026`               |
| Key pair | `unmimic-key`                          |

Planned scripts:

- `setup.sh` — install vLLM, Unsloth, TRL; clone the repo; install `.[ml,experiments]`.
- `autoshutdown.sh` — stop the instance after 30 minutes idle.
- `sync_s3.sh` — sync checkpoints and traces to the bucket.

Rules: the instance reaches S3 through an IAM instance role, never access keys. Never
point attack tests at Bedrock or any hosted model API.
