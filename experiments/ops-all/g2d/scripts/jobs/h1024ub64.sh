set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 1024 --configs configs/ops/unet-b64.json --out "$OUT/tr" --wall-seconds 3000
