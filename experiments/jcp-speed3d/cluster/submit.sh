#!/bin/bash
# jcp-speed3d: submit ONE staged attempt (lane cap 3 concurrent sp_ jobs). The staged copy goes to a private staging path;
# under an flock the remote side refuses if 3 sp_ jobs are live or the attempt directory exists, then moves staging into
# place and runs sbatch. squeue before and after (CLAUDE.md duplicate-submit rule). Pattern of jcp-combined.
set -euo pipefail
CM="${JCP_SSH_CM:-/tmp/jcpspeed-cm-%r@%h}"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=900 -o ConnectTimeout=30"
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-speed3d
LOCAL="$ROOT/experiments/jcp-speed3d/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
echo "--- squeue before"; $SSH tufts-login 'squeue -u $USER -h -o "%i %j %T"'
TAG="$A.$(hostname).$$.$(date +%s%N)"
$SSH tufts-login "mkdir -p $NS/.staging && mkdir $NS/.staging/$TAG"
rsync -e "$SSH" -a "$LOCAL/" "tufts-login:$NS/.staging/$TAG/"
$SSH tufts-login "cd $NS && flock -x -w 60 .submit.lock bash -c '
  set -euo pipefail
  q=\$(squeue -u \$USER -h -o \"%j\") || { echo \"squeue failed: refusing\"; exit 5; }
  n=\$(printf \"%s\\n\" \"\$q\" | grep -c \"^sp_\" || true)
  if [ \"\$n\" -ge 3 ]; then echo \"3 sp_ jobs live: refusing (lane cap 3)\"; exit 4; fi
  t=\$(printf \"%s\\n\" \"\$q\" | grep -c . || true)
  if [ \"\$t\" -ge 8 ]; then echo \"\$t jobs live on the account: refusing (campaign cap 8)\"; exit 6; fi
  test ! -e $NS/$A || { echo \"$NS/$A exists: refusing\"; exit 3; }
  mv $NS/.staging/$TAG $NS/$A
  cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch'"
echo "--- squeue after"; $SSH tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
