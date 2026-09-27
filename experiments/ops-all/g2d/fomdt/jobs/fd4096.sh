set -uo pipefail
cd experiments/burgers-bank-knob
"$PY" "$ROOT/fomdt/scripts/fomdt.py" --mesh 4096 --out "$OUT"
