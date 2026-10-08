#!/bin/bash
# jcp-smooth-bank: submit ONE staged attempt. Lane cap: ONE lane job in the queue in ANY state.
# A namespace lock (atomic mkdir on the shared file system) serialises check-through-submit; the attempt directory is
# created atomically (fails if it exists); squeue before and after (CLAUDE.md duplicate-submit rule).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcpsmooth
LOCAL="$ROOT/experiments/jcp-smooth-bank/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
ssh tufts-login "mkdir -p $NS && mkdir $NS/.submit.lock" || { echo "lock held: another submit in progress"; exit 5; }
trap 'ssh tufts-login "rmdir $NS/.submit.lock"' EXIT
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
NLANE=$(echo "$Q" | awk '$2 ~ /^jcps_/' | grep -c . || true)
echo "--- squeue before (lane jobs in queue, any state=$NLANE)"; echo "$Q"
[ "$NLANE" -lt 1 ] || { echo "lane cap reached"; exit 4; }
ssh tufts-login "mkdir $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
