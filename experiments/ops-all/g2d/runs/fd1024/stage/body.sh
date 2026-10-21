set -uo pipefail
cd experiments/burgers-bank-knob
"$PY" "$ROOT/fomdt/scripts/fomdt.py" --mesh 1024 --out "$OUT"
