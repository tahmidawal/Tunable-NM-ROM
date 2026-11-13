# Rebuttal push step 4, amendment A4: longer frozen-convection lag at the cheapest passing step (dev Re 95).
set -euo pipefail
"$PY" fe_ladder.py --manifest tasks/manifest_production.json --case re95 --out results/ladder \
  --entries '[[2,400,80],[2,400,160]]'
