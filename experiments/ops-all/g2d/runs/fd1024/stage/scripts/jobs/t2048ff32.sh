export BT_MESH=2048
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scripts/train_mem.py" --mesh 2048 --out "$OUT" --arms fno-w96f32 --wall-seconds 3000
