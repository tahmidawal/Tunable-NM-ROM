export BT_MESH=512
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scripts/train_mem.py" --mesh 512 --out "$OUT" --arms unet-large --wall-seconds 3000
