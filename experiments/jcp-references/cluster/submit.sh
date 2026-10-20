#!/bin/bash
# jcp-references: submit ONE staged attempt (A5.6). A local flock serialises submissions from this lane; the remote
# job directory is created with an exclusive mkdir (fails if it exists); refuses if jr_<attempt> is already queued or
# the lane already has 4 jobs queued/running (concurrency cap); squeue before and after. The only extra sbatch flag
# accepted is a dependency: $2 = --dependency=afterok:<jobid>[:<jobid>...].
set -euo pipefail
A="$1"; EXTRA="${2:-}"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
if [ -n "$EXTRA" ] && ! [[ "$EXTRA" =~ ^--dependency=afterok(:[0-9]+)+$ ]]; then echo "refusing extra flags: $EXTRA"; exit 6; fi
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-ref
LOCAL="$ROOT/experiments/jcp-references/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
exec 9>"$ROOT/experiments/jcp-references/runs/.submit.lock"
flock -x 9
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "jr_$A"; then echo "jr_$A already queued: refusing"; exit 3; fi
N=$(echo "$Q" | awk '{print $2}' | grep -c '^jr_' || true)
if [ "$N" -ge 4 ]; then echo "lane cap: $N jr_ jobs already queued/running: refusing"; exit 5; fi
ssh tufts-login "df -h /cluster/tufts/paralab | tail -1; mkdir -p $NS && mkdir $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch $EXTRA run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
