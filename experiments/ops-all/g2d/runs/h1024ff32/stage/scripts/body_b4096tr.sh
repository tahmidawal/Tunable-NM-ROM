# g2d Burgers 4096^2 operator training attempt: data generated once into host memory, four networks in sequence
set -uo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
cd experiments/burgers-compare-hires
"$PY" "$ROOT/scripts/train_mem.py" --mesh 4096 --out "$OUT" --arms fno-large,unet-refine,tsol-refine,don-small --wall-seconds 3000
