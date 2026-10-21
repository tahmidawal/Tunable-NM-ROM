set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 512 --configs configs/ops/unet-large.json --out "$OUT/tr" --wall-seconds 3000
