set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 256 --configs configs/ops/fno.json --out "$OUT/tr" --wall-seconds 3000
