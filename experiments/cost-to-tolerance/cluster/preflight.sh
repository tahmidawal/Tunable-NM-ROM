#!/usr/bin/env bash
# preflight.sh <mode>        mode: panels | fom | ceiling | recover
#
# Run the mode's code path END TO END locally, at a trivial size, BEFORE submitting
# it to the cluster.  Not a correctness check on the physics -- a check that every
# line the cluster job will execute actually executes, especially the RECORD-APPEND
# paths that arithmetic-only testing never reaches.
#
# WHY.  Two cluster jobs in this cell died to defects a 30-second local run would
# have caught, and neither was in the measurement:
#   * ctol_p_n512  -- an unchunked jacfwd in the ceiling OOMed at 16.87 GiB, 50 min in;
#   * ctol_ceil_all -- `dict() got multiple values for keyword argument 'k'`, because
#     the new spectrum payload carried a key that collided with the row it was
#     splatted into.  Died after 24 s having done all the arithmetic correctly.
# Every defect in this cell so far has been in the bookkeeping AROUND the
# measurement, so that is what this exercises.
set -euo pipefail
MODE="${1:?usage: preflight.sh <panels|fom|ceiling|recover>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL="$(dirname "$HERE")"
PY=/home/tahmid/Dev/.venv/bin/python
OUT="$CELL/runs/smoke/preflight_$MODE.json"
mkdir -p "$CELL/runs/smoke"
rm -f "$OUT"
# shellcheck disable=SC1091
source /etc/profile.d/jax-mem.sh

# trivial-but-complete: one mesh, one or two k, one tau, 2 sources, tiny EQ
COMMON="PKL_DIR=$CELL/ckpt_poisson NS=32 N_TEST=2 TAUS=1e-2 M=32 MQ=64 M_BIG=64 K_BIG=32 \
MQ_4M=128 CTOL_CAND_CAP=512 CTOL_EQ_SNAPS=8 CTOL_EQ_ROWS=512 TIME_REPS=1 TIME_WARM=1 \
BURN_IN_S=0.2 FOM_LADDER=1e-2,1e-13 GN_ITERS=40 CEIL_BUDGET=40 CEIL_CHUNK=4096"

case "$MODE" in
  panels)  ENV="$COMMON KS=8,32 DO_SUPP=1 POOL_CONTROL=1 CAP_CONTROL=1 CAP_CONTROL_MAX=1024 DO_CEILING=1" ;;
  fom)     ENV="$COMMON KS=8 FOM_ONLY=1 DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_CEILING=0" ;;
  ceiling) ENV="$COMMON KS=8 CEILING_ONLY=1 DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0" ;;
  recover) ENV="$COMMON KS=8,32 DO_SUPP=0 POOL_CONTROL=0 CAP_CONTROL=0 DO_CEILING=0 DO_POD_DIRECT=0" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

echo "preflight[$MODE]: running the real code path locally at N=32 ..."
cd "$CELL"
# shellcheck disable=SC2086
env $ENV JAX_DEFAULT_MATMUL_PRECISION=highest \
  timeout 900 jaxrun "$PY" -u ctol_poisson.py "$OUT" > "$CELL/runs/smoke/preflight_$MODE.log" 2>&1 || {
    echo "preflight[$MODE] FAILED -- do NOT submit.  Tail:" >&2
    tail -25 "$CELL/runs/smoke/preflight_$MODE.log" >&2
    exit 1
  }
# the JSON must exist, parse, and contain at least one record of the kind this mode writes
"$PY" - "$OUT" "$MODE" <<'EOF'
import json, sys
out, mode = sys.argv[1], sys.argv[2]
d = json.load(open(out))
if mode == "fom":
    n = len(d.get("fom", [])); kind = "fom rungs"
elif mode == "ceiling":
    n = len([x for x in d.get("supplementary", []) if x.get("method") == "oracle_ceiling"])
    kind = "ceiling records"
else:
    n = len(d.get("rows", [])); kind = "ROM rows"
if n == 0:
    raise SystemExit(f"preflight[{mode}] wrote NO {kind} -- the record-append path did "
                     f"not execute; do NOT submit")
print(f"preflight[{mode}]: OK -- {n} {kind} written and parsed")
EOF
