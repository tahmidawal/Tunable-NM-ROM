# Rebuttal push step 4, amendment A3: frozen-convection lag variants of the cheapest passing step (dev Re 95).
set -euo pipefail
"$PY" fe_ladder.py --manifest tasks/manifest_production.json --case re95 --out results/ladder \
  --entries '[[2,400,10],[2,400,20],[2,400,40],[2,400,100000]]'
