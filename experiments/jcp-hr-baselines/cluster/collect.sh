#!/bin/bash
# jcp-hr-baselines: pull one finished attempt (logs + output), verify OUTPUTS.sha256 locally, record sacct, then (only
# after a successful verification and when called with --delete) remove the cluster job directory.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-hrb
LOCAL="$ROOT/experiments/jcp-hr-baselines/runs/$A"
SSH="ssh -o ConnectTimeout=30"
df -h /home | tail -1
mkdir -p "$LOCAL/archive"
timeout 3600 rsync -e "$SSH" -a "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/output" "tufts-login:$NS/$A/OUTPUTS.sha256" "$LOCAL/archive/" || true
( cd "$LOCAL/archive" && sha256sum -c OUTPUTS.sha256 --quiet && echo "checksums OK" )
JOB=$(ls "$LOCAL/archive/logs" | sed -n 's/\.out$//p' | head -1)
timeout 60 $SSH tufts-login "sacct -j $JOB --format=JobID,State,Elapsed,AllocTRES%60,ExitCode -P" > "$LOCAL/archive/sacct.txt"
cat "$LOCAL/archive/sacct.txt"
if [ "${2:-}" = "--delete" ]; then timeout 120 $SSH tufts-login "rm -rf $NS/$A" && echo "deleted $NS/$A"; fi
