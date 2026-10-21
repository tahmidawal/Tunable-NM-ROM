export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
set -uo pipefail
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scaling256/scripts/train_scale.py" --arm fno-w96f32 --n-train 2048 --out "$OUT" --wall-seconds 7200
