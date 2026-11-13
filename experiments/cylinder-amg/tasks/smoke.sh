# Smoke (plumbing only, development Re 95, 0.2 s = 60 steps, two refreshes): every solver kind runs and is scored.
set -euo pipefail
"$PY" fe_amg.py --manifest tasks/manifest_production.json --case re95 --meshes tasks/MESHES.json \
  --entries-file tasks/entries_smoke.json --out results/smoke --reps 1 --duration 0.2
