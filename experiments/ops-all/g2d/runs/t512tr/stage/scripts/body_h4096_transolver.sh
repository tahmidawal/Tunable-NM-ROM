# g2d heat 4096^2 operator training, one network (transolver), heat-compare-hires train_ops.py unchanged
set -uo pipefail
cd experiments/heat-compare-hires
"$PY" ops/train_ops.py --mesh 4096 --configs configs/ops/transolver.json --out "$OUT/tr" --wall-seconds 3000
