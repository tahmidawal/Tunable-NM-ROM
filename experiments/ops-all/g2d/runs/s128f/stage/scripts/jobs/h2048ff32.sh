set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 2048 --configs configs/ops/fno-w96f32.json --out "$OUT/tr" --wall-seconds 3000
