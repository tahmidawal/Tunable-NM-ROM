#!/bin/bash
# quadrature-burgers3d: submit ONE staged attempt. Refuses if q3_<attempt> is already queued or the remote dir exists;
# rsyncs runs/<attempt>/ to its own namespace dir; squeue before and after. Extra sbatch flags (e.g. --dependency) via $2.
set -euo pipefail
A="$1"; EXTRA="${2:-}"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/quad3d_20261001
LOCAL="$ROOT/experiments/quadrature-burgers3d/runs/$A"
[ -f "$LOCAL/run.sbatch" ]
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%i %j %T"')
echo "--- squeue before"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "q3_$A"; then echo "q3_$A already queued: refusing"; exit 3; fi
ssh tufts-login "df -h /cluster/tufts/paralab | tail -1; test ! -e $NS/$A && mkdir -p $NS/$A"
rsync -a "$LOCAL/" "tufts-login:$NS/$A/"
ssh tufts-login "cd $NS/$A && sha256sum -c MANIFEST.sha256 --quiet && sbatch $EXTRA run.sbatch"
echo "--- squeue after"; ssh tufts-login 'squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"'
