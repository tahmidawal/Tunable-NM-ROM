#!/bin/bash
# Submit only after independent review of this exact committed stage.
set -euo pipefail
ATTEMPT="$1"
[[ "$ATTEMPT" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/b3d-recon-20261027
STAGE="$ROOT/experiments/burgers-recon3d/runs/$ATTEMPT"
[ -f "$STAGE/run.sbatch" ]
ssh tufts-login 'squeue -u $USER -o "%.18i %.28j %.8T %.10M %R"'
ssh tufts-login "mkdir -p $NS && mkdir $NS/.submit.lock"
trap 'ssh tufts-login "rmdir $NS/.submit.lock"' EXIT
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%j"')
if printf '%s\n' "$Q" | grep -q '^br3_'; then echo 'Own lane already has an active allocation'; exit 3; fi
ssh tufts-login "test ! -e $NS/$ATTEMPT && mkdir $NS/$ATTEMPT"
scp -r "$STAGE/." "tufts-login:$NS/$ATTEMPT/"
ssh tufts-login "cd $NS/$ATTEMPT && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
ssh tufts-login 'squeue -u $USER -o "%.18i %.28j %.8T %.10M %R"'
