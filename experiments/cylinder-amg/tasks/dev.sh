# PROTOCOL.md development job: Re 95, full 1.7 s window, one run each (also timed once): lu, amg1/amg2 at the rtol ladder,
# amg1 with Ruge-Stueben, amg1 with the mass-only Schur approximation.
set -euo pipefail
"$PY" fe_amg.py --manifest tasks/manifest_production.json --case re95 --meshes tasks/MESHES.json \
  --entries-file tasks/entries_dev.json --out results/dev --reps 1
