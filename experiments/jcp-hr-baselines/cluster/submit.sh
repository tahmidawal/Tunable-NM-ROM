#!/bin/bash
# jcp-hr-baselines: submit ONE staged attempt. Lane cap 2 live hrb_ jobs; campaign cap 8 live jobs on the account.
# Staged copy goes to a private staging path, then under an flock the remote side refuses if a cap would be exceeded
# or the attempt directory exists, moves staging into place and runs sbatch. squeue before and after (duplicate-submit rule).
set -euo pipefail
CM="${HRB_SSH_CM:-/tmp/hrb-cm-%r@%h}"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=900 -o ConnectTimeout=30"
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-hrb
LOCAL="$ROOT/experiments/jcp-hr-baselines/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
# job audit 1: the bundle is exactly its manifest (every file listed and matching; nothing else)
( cd "$LOCAL" && sha256sum -c MANIFEST.sha256 --quiet && \
  [ -z "$(comm -3 <(find . ! -type d ! -path ./MANIFEST.sha256 | sed 's|^\./||' | sort) <(awk '{print $2}' MANIFEST.sha256 | sort))" ] ) \
  || { echo "bundle differs from its manifest: refusing"; exit 8; }
# A34.2: the attempt must hold exactly one unbound reservation (marked 'submitting' under the ledger lock)
/home/tahmid/Dev/.venv/bin/python "$ROOT/experiments/jcp-hr-baselines/cluster/ledger.py" submit-check "$A"
echo "--- squeue before"; timeout 60 $SSH tufts-login 'squeue -u $USER -h -o "%i %j %T"'
TAG="$A.$(hostname).$$.$(date +%s%N)"
timeout 60 $SSH tufts-login "mkdir -p $NS/.staging && mkdir $NS/.staging/$TAG"
timeout 1800 rsync -e "$SSH" -a "$LOCAL/" "tufts-login:$NS/.staging/$TAG/"
# COMMON-BRIEF 14: every sbatch under the shared campaign lock; account count (queued + running) < 8 inside it
timeout 900 $SSH tufts-login "cd $NS && flock -x -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c '
  set -euo pipefail
  q=\$(squeue -u \$USER -h -o \"%j\") || { echo \"squeue failed: refusing\"; exit 5; }
  n=\$(printf \"%s\\n\" \"\$q\" | grep -c \"^hrb_\" || true)
  t=\$(printf \"%s\\n\" \"\$q\" | grep -c . || true)
  if [ \"\$n\" -ge 2 ]; then echo \"2 hrb_ jobs live: refusing (lane cap 2)\"; exit 4; fi
  if [ \"\$t\" -ge 8 ]; then echo \"8 jobs live on the account: refusing (campaign cap 8)\"; exit 75; fi
  test ! -e $NS/$A || { echo \"$NS/$A exists: refusing\"; exit 3; }
  mv $NS/.staging/$TAG $NS/$A
  cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch --no-requeue run.sbatch'" | tee "$LOCAL.SUBMIT.out"
JID=$(sed -n 's/^Submitted batch job \([0-9]*\)$/\1/p' "$LOCAL.SUBMIT.out")
[ -n "$JID" ] && /home/tahmid/Dev/.venv/bin/python "$ROOT/experiments/jcp-hr-baselines/cluster/ledger.py" bind "$A" "$JID"
echo "--- squeue after"; timeout 60 $SSH tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
