#!/bin/bash
# fp64-query (copy of b2d-query-speed cluster/pull.sh, namespace changed): pull output + logs of a finished attempt into results/<attempt>, verify, then delete the cluster dir.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd); NS=/cluster/tufts/paralab/tawal01/fp64-query-20261115
ssh tufts-login "grep -q ALL-DONE $NS/$A/logs/*.out" || { echo "not complete: kept on cluster"; KEEP=1; }
mkdir -p "$LANE/results/$A"
rsync -a --exclude='*.npy' --exclude='*.npz' "tufts-login:$NS/$A/output/" "$LANE/results/$A/" || true
rsync -a "tufts-login:$NS/$A/logs/" "$LANE/results/$A/logs/"
rsync -a "tufts-login:$NS/$A/COMMIT.txt" "$LANE/results/$A/"
grep -h jax_backend "$LANE/results/$A"/logs/*.out
[ -n "${KEEP:-}" ] || ssh tufts-login "rm -rf $NS/$A"
ls "$LANE/results/$A"
