#!/bin/bash
# usage: cluster/submit.sh <job>   (after cluster/stage.py). One job per directory; squeue printed before
# and after; refuses if the remote directory exists or the group already has >= CAP jobs queued/running.
set -euo pipefail
J=$1
CAP=${CAP:-2}
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/p2d
R=$HERE/runs/$J
test -f "$R/run.sbatch" && test ! -f "$R/JOBID.txt"
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 tufts-login "squeue -u tawal01 -o '%.10i %.22j %.8T %.10M %R'" > "$R/QUEUE-BEFORE.log"
MINE=$(grep -c ' p2d_' "$R/QUEUE-BEFORE.log" || true)
if grep -q " p2d_$J " "$R/QUEUE-BEFORE.log"; then echo "duplicate: p2d_$J already queued"; exit 4; fi
[ "$MINE" -lt "$CAP" ] || { echo "p2d already has $MINE jobs (cap $CAP)"; exit 3; }
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 tufts-login "mkdir -p $NS && test ! -e $NS/$J"
rsync -a --exclude QUEUE-BEFORE.log "$R/" "tufts-login:$NS/$J/"
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 tufts-login "cd $NS/$J && sha256sum -c MANIFEST.sha256 --quiet && sbatch --parsable run.sbatch" | tee "$R/JOBID.txt"
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 tufts-login "squeue -u tawal01 -o '%.10i %.22j %.8T %.10M %R'" > "$R/QUEUE-AFTER.log"
N=$(grep -c " p2d_$J " "$R/QUEUE-AFTER.log" || true)
[ "$N" -le 1 ] || { echo "WARNING: $N jobs named p2d_$J"; exit 5; }
echo "submitted $J -> $(cat "$R/JOBID.txt")"
