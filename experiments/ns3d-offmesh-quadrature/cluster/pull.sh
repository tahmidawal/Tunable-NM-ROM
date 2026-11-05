#!/bin/bash
# Pull one job with checksums, check preflight/exit/status, then delete the remote dir (kept if unclean
# unless FORCE_DELETE=1). Usage: pull.sh <job> <jobid>
set -euo pipefail
JOB="$1"; ID="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC=/cluster/tufts/paralab/tawal01/ns3d-offmesh-q-20261105/$JOB
DEST="$HERE/experiments/ns3d-offmesh-quadrature/runs/$JOB"
mkdir -p "$DEST"
rsync -az "tufts-login:$SRC/output/" "$DEST/output/"
rsync -az "tufts-login:$SRC/OUTPUTS.sha256" "tufts-login:$SRC/COMMIT.txt" "$DEST/"
rsync -az "tufts-login:$SRC/logs/" "$DEST/logs/"
( cd "$DEST" && sha256sum -c --quiet OUTPUTS.sha256 ) && echo "checksums verified"
LOG="$DEST/logs/$ID.out"
OK=1
grep -q "jax_backend=gpu" "$LOG" || { echo "no jax_backend=gpu"; OK=0; }
grep -q "JAX_DEFAULT_MATMUL_PRECISION=highest" "$LOG" || { echo "precision not highest"; OK=0; }
grep -q "RUN_EXIT=0" "$LOG" || OK=0
/home/tahmid/Dev/.venv/bin/python -c "import json,sys; s=json.load(open(sys.argv[1])); print('status',s.get('status'),'failures',s.get('failures')); sys.exit(0 if s.get('status')=='final' else 1)" "$DEST/output/summary.json" || OK=0
grep -iE "out of memory|RESOURCE_EXHAUSTED|constant folding|captured" "$DEST/logs/$ID.err" | head -5 || true
echo "clean=$OK"
if [[ "$OK" -eq 1 || "${FORCE_DELETE:-0}" -eq 1 ]]; then ssh tufts-login "rm -rf $SRC"; echo "remote $SRC deleted"
else echo "remote $SRC KEPT"; fi
