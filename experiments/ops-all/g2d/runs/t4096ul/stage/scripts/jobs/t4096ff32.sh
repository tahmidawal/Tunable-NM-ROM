export BT_MESH=4096
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scripts/train_mem.py" --mesh 4096 --out "$OUT" --arms fno-w96f32 --wall-seconds 3000
