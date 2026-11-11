# Rebuttal push step 4 (PROTOCOL.md): FE ladder on development Re 95, cheapest entries first.
set -euo pipefail
"$PY" fe_ladder.py --manifest tasks/manifest_production.json --case re95 --out results/ladder \
  --entries '[[2,200,2],[2,400,5],[2,800,10],[3,200,2],[3,400,5],[3,800,10],[2,1600,20]]'
