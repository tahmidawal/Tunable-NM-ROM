# PROTOCOL.md held-out job: Re 105 opened once for this lane. Entries (tasks/entries_heldout.json, written from the
# development selection): lu, amg1 (selected rtol), amg2 (selected rtol), lu again (lu_b, drift check); each one scored
# run + 5 timed repetitions in total (run0 is scored and is the first repetition, exactly as fe_ladder2 --reps 5).
set -euo pipefail
"$PY" fe_amg.py --manifest tasks/manifest_production.json --case re105 --meshes tasks/MESHES.json \
  --entries-file tasks/entries_heldout.json --out results/heldout --reps 5 --allow-heldout
