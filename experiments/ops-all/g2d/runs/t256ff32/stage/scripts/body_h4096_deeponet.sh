# g2d heat 4096^2 operator training, one network (deeponet), heat-compare-hires train_ops.py unchanged
set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 4096 --configs configs/ops/deeponet.json --out "$OUT/tr" --wall-seconds 3000
