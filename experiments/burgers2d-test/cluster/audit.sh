#!/bin/bash
# burgers2d-test: audit one collected attempt (runs/<attempt>/archive) with the staged inputs of that attempt.
set -euo pipefail
A="$1"; [[ "$A" =~ ^t[0-9]+$ ]]
LANE=$(cd "$(dirname "$0")/.." && pwd)
S="$LANE/runs/$A"; ARC="$S/archive"
L=${A#t}
case $L in
  256|512|1024) PRIOR="--prior $LANE/../burgers2d-speed/checks/h$L-summary.json";;
  4096) PRIOR="--prior $LANE/checks/prior-bkh64-summary-340627ca.json";;
  *) PRIOR="";;
esac
/home/tahmid/Dev/.venv/bin/python "$LANE/audit_b2test.py" "$ARC" \
  --checkpoint "$S/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" \
  --rotation "$S/experiments/burgers-bank-knob/inputs/rotation_R512.npz" \
  --directions "$S/experiments/b-panel/inputs/directions_qtd02.npz" \
  --out "$LANE/checks/$A-summary.json" $PRIOR 2>&1 | tee "$LANE/checks/$A-audit.log"
