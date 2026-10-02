#!/bin/sh
set -e
PY=.venv/bin/python
for pde in "$@"; do
  $PY scripts/gen_data.py $pde --neval 6 --dt_ref 0.0025
  $PY scripts/train.py $pde
  $PY scripts/quad_study.py $pde
  $PY scripts/rollout_study.py $pde
done
$PY scripts/report.py "$@"
