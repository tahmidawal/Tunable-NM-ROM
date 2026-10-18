#!/bin/bash
# jcp-hr-baselines: pull one TERMINAL attempt (logs + output), verify a freshly computed remote manifest covering outputs
# AND logs, record sacct; only with --delete, and only after every step succeeded, remove the cluster job directory.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-hrb
LOCAL="$ROOT/experiments/jcp-hr-baselines/runs/$A"
SSH="ssh -o ConnectTimeout=30"
df -h /home | tail -1
JOB=$(timeout 60 $SSH tufts-login "ls $NS/$A/logs" | sed -n 's/\.out$//p' | head -1)
[ -n "$JOB" ]
# terminal only if sacct (queried successfully) reports a terminal state for the job allocation (fail-closed)
ST=$(timeout 60 $SSH tufts-login "sacct -j $JOB -X -n -P --format=State" | head -1 | awk '{print $1}')
case "$ST" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) echo "job $JOB terminal: $ST";;
  *) echo "job $JOB state '$ST' is not terminal or unknown: refusing"; exit 3;; esac
timeout 300 $SSH tufts-login "bash -o pipefail -euc 'cd $NS/$A && find logs output -type f -print0 | sort -z | xargs -0 sha256sum'" > "$LOCAL/REMOTE.sha256.new"
mkdir -p "$LOCAL/archive"
timeout 3600 rsync -e "$SSH" -a "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/output" "$LOCAL/archive/"
( cd "$LOCAL/archive" && sha256sum -c "../REMOTE.sha256.new" --quiet )
mv "$LOCAL/REMOTE.sha256.new" "$LOCAL/REMOTE.sha256"
echo "transfer verified against the fresh remote manifest ($(wc -l < "$LOCAL/REMOTE.sha256") files)"
timeout 60 $SSH tufts-login "sacct -j $JOB --format=JobID,State,Elapsed,AllocTRES%60,ExitCode -P" > "$LOCAL/archive/sacct.txt"
cat "$LOCAL/archive/sacct.txt"
if [ "${2:-}" = "--delete" ]; then timeout 120 $SSH tufts-login "rm -rf $NS/$A" && echo "deleted $NS/$A"; fi
