export BT_MESH=1024
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scripts/train_mem.py" --mesh 1024 --out "$OUT" --arms tsol-large --wall-seconds 3000
