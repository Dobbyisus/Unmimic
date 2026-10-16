#!/usr/bin/env bash
# Manage the Unmimic GPU machine from AWS CloudShell (already signed in; no access keys).
#
# Upload this file and bootstrap.sh to CloudShell (Actions > Upload file), then:
#
#   bash unmimic-aws.sh setup <your-ip>   one-time: IAM role for S3, SSH security group
#   bash unmimic-aws.sh capacity          spot availability score and price per GPU type/zone
#   bash unmimic-aws.sh launch            create a g5.xlarge spot machine (runs bootstrap.sh)
#   bash unmimic-aws.sh launch --on-demand    same, on-demand (always available, ~2.5x price)
#   TYPE=g6.xlarge bash unmimic-aws.sh launch use another 24 GB GPU type (g6 = NVIDIA L4)
#   bash unmimic-aws.sh status            state, public IP, and the commands to connect
#   bash unmimic-aws.sh stop | start      stopped = no GPU charge (the disk is still billed)
#   bash unmimic-aws.sh allow-ip <ip>     your IP changed: move the SSH rule to the new one
#   bash unmimic-aws.sh requests          list spot requests; cancel-requests cancels open ones
#   bash unmimic-aws.sh terminate         delete the machine and its disk
#
# Your IP: run on your LAPTOP (not in CloudShell): curl -s https://checkip.amazonaws.com
set -euo pipefail

REGION=us-east-1
BUCKET=unmimic-dobbyisus-2026
KEY_NAME=unmimic-key
ROLE=unmimic-ec2-role
PROFILE=unmimic-ec2-profile
SG_NAME=unmimic-ssh
NAME=${NAME:-unmimic-gpu}
TYPE=${TYPE:-g5.xlarge}
# 24 GB GPUs within the 8-vCPU G quota, compared by `capacity`.
CANDIDATE_TYPES="g5.xlarge g6.xlarge"
DISK_GB=150
AMI_PATTERN='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*'

# tr: the Windows AWS CLI ends lines with \r, which would end up inside parsed values.
aws() { command aws --region "$REGION" "$@" | tr -d '\r'; }
die() { echo "error: $*" >&2; exit 1; }
none_to_empty() { [ "$1" = None ] && echo "" || echo "$1"; }

# An IP, or a range like 152.58.0.0/16: mobile carriers (e.g. Jio) send SSH out through
# a different public IP than web traffic, so a single IP may not match.
check_ip() {
  [[ ${1:-} =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(/[0-9]{1,2})?$ ]] ||
    die "give your IPv4 address or range (on your laptop: curl -s https://checkip.amazonaws.com)"
}

as_cidr() { [[ $1 == */* ]] && echo "$1" || echo "$1/32"; }

sg_id() {
  none_to_empty "$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text)"
}

instance_id() {
  none_to_empty "$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text)"
}

s3_policy() {
  cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::$BUCKET"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::$BUCKET/*"}
  ]
}
EOF
}

set_ssh_rule() {
  local sg=$1 ip=$2 cidr
  for cidr in $(aws ec2 describe-security-groups --group-ids "$sg" \
      --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text); do
    aws ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$cidr" >/dev/null
  done
  aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
    --cidr "$(as_cidr "$ip")" >/dev/null
  echo "SSH allowed only from $(as_cidr "$ip")"
}

cmd_setup() {
  check_ip "${1:-}"
  if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
      >/dev/null
    echo "created IAM role $ROLE"
  fi
  aws iam put-role-policy --role-name "$ROLE" --policy-name unmimic-s3 --policy-document "$(s3_policy)"
  echo "role $ROLE may use bucket $BUCKET only"
  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
    echo "created instance profile $PROFILE (wait ~15 seconds before launch)"
  fi

  local vpc sg
  vpc=$(none_to_empty "$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)")
  [ -n "$vpc" ] || die "no default VPC in $REGION"
  sg=$(sg_id)
  if [ -z "$sg" ]; then
    sg=$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$vpc" \
      --description "SSH to the Unmimic GPU machine" --query GroupId --output text)
    echo "created security group $sg"
  fi
  set_ssh_rule "$sg" "$1"
  aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1 ||
    die "key pair $KEY_NAME not found in $REGION"
  echo "setup done; next: bash $0 launch"
}

cmd_allow_ip() {
  check_ip "${1:-}"
  local sg; sg=$(sg_id)
  [ -n "$sg" ] || die "run setup first"
  set_ssh_rule "$sg" "$1"
}

cmd_launch() {
  local market=(--instance-market-options
    '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}')
  local kind="spot"
  if [ "${1:-}" = --on-demand ]; then
    market=()
    kind="on-demand"
  fi
  local existing; existing=$(instance_id)
  [ -z "$existing" ] || die "machine $existing already exists; use: bash $0 status / start"
  local here; here=$(cd "$(dirname "$0")" && pwd)
  [ -f "$here/bootstrap.sh" ] || die "upload bootstrap.sh next to this script"
  local sg; sg=$(sg_id)
  [ -n "$sg" ] || die "run setup first"
  local user_data="$here/bootstrap.sh"
  # Git Bash on Windows: the Windows AWS CLI needs C:/... paths, not /c/...
  if command -v cygpath >/dev/null; then user_data=$(cygpath -m "$user_data"); fi

  local ami root
  read -r ami root < <(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=$AMI_PATTERN" Name=state,Values=available \
    --query 'sort_by(Images,&CreationDate)[-1].[ImageId,RootDeviceName]' --output text)
  [ -n "$ami" ] && [ "$ami" != None ] || die "Deep Learning AMI not found"
  echo "AMI $ami; launching $kind $TYPE"

  local subnet zone id="" err code
  err=$(mktemp)
  while read -r subnet zone; do
    if id=$(aws ec2 run-instances \
        --image-id "$ami" --instance-type "$TYPE" --key-name "$KEY_NAME" \
        --subnet-id "$subnet" --security-group-ids "$sg" \
        --iam-instance-profile "Name=$PROFILE" \
        "${market[@]}" \
        --block-device-mappings \
          "[{\"DeviceName\":\"$root\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --metadata-options HttpTokens=required \
        --user-data "file://$user_data" \
        --tag-specifications \
          "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=unmimic}]" \
          "ResourceType=volume,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=unmimic}]" \
        --query 'Instances[0].InstanceId' --output text 2>"$err"); then
      break
    fi
    id=""
    code=$(grep -oE '\(([A-Za-z.]+)\)' "$err" | head -1 | tr -d '()')
    case "$code" in
      InsufficientInstanceCapacity | Unsupported | SpotMaxPriceTooLow)
        echo "$zone: $code, trying the next zone"
        continue ;;
    esac
    cat "$err" >&2
    die "launch failed in $zone (${code:-unknown error})"
  done < <(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
    --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text)
  [ -n "$id" ] || die "no $kind $TYPE capacity in any zone. Try: bash $0 capacity, then
  TYPE=g6.xlarge bash $0 launch, or later, or: bash $0 launch --on-demand"
  echo "launched $id; waiting until it is running..."
  aws ec2 wait instance-running --instance-ids "$id"
  cmd_status
  echo
  echo "First boot installs vLLM (~10 minutes). Done when ~/BOOTSTRAP_DONE exists on the machine."
}

cmd_capacity() {
  echo "Spot placement score per zone (1 = unlikely, 10 = very likely to launch):"
  local type
  for type in $CANDIDATE_TYPES; do
    aws ec2 get-spot-placement-scores --instance-types "$type" --target-capacity 1 \
      --region-names "$REGION" --single-availability-zone \
      --query 'SpotPlacementScores[].[AvailabilityZoneId,Score]' --output text |
      sort -k2 -nr | sed "s/^/  $type  /"
  done
  echo "Current spot price, USD/hour (on-demand: g5.xlarge ~1.01, g6.xlarge ~0.80):"
  # shellcheck disable=SC2086
  aws ec2 describe-spot-price-history --instance-types $CANDIDATE_TYPES \
    --product-descriptions Linux/UNIX --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query 'SpotPriceHistory[].[InstanceType,AvailabilityZone,SpotPrice]' --output text |
    sort | sed 's/^/  /'
}

cmd_requests() {
  aws ec2 describe-spot-instance-requests \
    --query 'SpotInstanceRequests[].[SpotInstanceRequestId,State,Status.Code,InstanceId,LaunchSpecification.InstanceType]' \
    --output table
}

cmd_cancel_requests() {
  local ids
  ids=$(aws ec2 describe-spot-instance-requests --filters Name=state,Values=open \
    --query 'SpotInstanceRequests[].SpotInstanceRequestId' --output text)
  if [ -z "$ids" ] || [ "$ids" = None ]; then
    echo "no open spot requests"
    return
  fi
  # shellcheck disable=SC2086
  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $ids >/dev/null
  echo "cancelled open spot requests: $ids"
}

cmd_status() {
  local id; id=$(instance_id)
  [ -n "$id" ] || { echo "no $NAME machine"; return; }
  local state ip zone
  read -r state ip zone < <(aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,Placement.AvailabilityZone]' \
    --output text)
  echo "machine $id: $state in $zone, public IP $ip"
  if [ "$state" = running ]; then
    echo "From your laptop (Git Bash, in the unmimic folder):"
    echo "  bash infra/aws/push_code.sh $ip"
    echo "  ssh -i ~/.ssh/unmimic-key.pem ubuntu@$ip"
  fi
}

cmd_stop() {
  local id; id=$(instance_id)
  [ -n "$id" ] || die "no $NAME machine"
  aws ec2 stop-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-stopped --instance-ids "$id"
  echo "stopped $id (no GPU charge; the ${DISK_GB} GB disk is still billed)"
}

cmd_start() {
  local id; id=$(instance_id)
  [ -n "$id" ] || die "no $NAME machine; use launch"
  aws ec2 start-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-running --instance-ids "$id"
  cmd_status
}

cmd_terminate() {
  local id; id=$(instance_id)
  [ -n "$id" ] || die "no $NAME machine"
  read -r -p "Delete $id and its disk (anything not synced to S3 is lost)? Type yes: " answer
  [ "$answer" = yes ] || die "cancelled"
  local request
  request=$(none_to_empty "$(aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)")
  # Cancel the persistent spot request first, or AWS starts a replacement machine.
  [ -z "$request" ] || aws ec2 cancel-spot-instance-requests --spot-instance-request-ids "$request" >/dev/null
  aws ec2 terminate-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$id"
  echo "terminated $id"
}

case "${1:-}" in
  setup) cmd_setup "${2:-}" ;;
  allow-ip) cmd_allow_ip "${2:-}" ;;
  launch) cmd_launch "${2:-}" ;;
  capacity) cmd_capacity ;;
  requests) cmd_requests ;;
  cancel-requests) cmd_cancel_requests ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  start) cmd_start ;;
  terminate) cmd_terminate ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
