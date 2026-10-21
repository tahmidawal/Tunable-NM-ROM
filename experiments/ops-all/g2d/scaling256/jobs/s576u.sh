export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scaling256/scripts/train_scale.py" --arm unet-b64 --n-train 576 --out "$OUT" --wall-seconds 7200
