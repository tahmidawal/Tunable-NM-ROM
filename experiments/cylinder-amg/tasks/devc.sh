# PROTOCOL.md development job, part c (split of tasks/entries_dev.json for wall time, see amendment D1): Re 95, full window, one run each.
set -euo pipefail
"$PY" fe_amg.py --manifest tasks/manifest_production.json --case re95 --meshes tasks/MESHES.json \
  --entries-file tasks/entries_devc.json --out results/dev --reps 1
