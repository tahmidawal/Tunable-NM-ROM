#!/bin/bash
# jcp-time2: submit ONE staged attempt (lane cap 1). The staged directory is copied to a private staging path first;
# then, under an flock on the namespace's lock file (atomic across concurrent invocations), the remote side refuses if
# any t2_ job is live in any state or the attempt directory exists, moves staging into place and runs sbatch.
# squeue before and after (CLAUDE.md duplicate-submit rule).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcptime2
LOCAL="$ROOT/experiments/jcp-time2/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
echo "--- squeue before"; ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"'
TAG="$A.$(hostname).$$.$(date +%s%N)"
ssh tufts-login "mkdir -p $NS/.staging && mkdir $NS/.staging/$TAG"
rsync -a "$LOCAL/" "tufts-login:$NS/.staging/$TAG/"
ssh tufts-login "cd $NS && flock -x -w 60 .submit.lock bash -c '
  set -euo pipefail
  if squeue -u \$USER -h -o \"%j\" | grep -q \"^t2_\"; then echo \"a t2_ job is live: refusing (lane cap 1)\"; exit 4; fi
  test ! -e $NS/$A || { echo \"$NS/$A exists: refusing\"; exit 3; }
  mv $NS/.staging/$TAG $NS/$A
  cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch'"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
