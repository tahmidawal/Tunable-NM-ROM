# Rebuttal push step 1: diagnosis of the existing r16 bank (r3-train01) against POD r16, development Re 95.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=$NS/r2-export01/results/export; TR=$NS/r3-train01/results
sha256sum -c inputs/R3-SHA256SUMS
"$PY" bank_diag.py --export "$EX" --manifest tasks/manifest_production.json --case re95 --bank-eval $TR/train/trained_bank_eval.npz \
  --bank-proj $TR/proj/proj_re95_trained.npz --rank 16 --out results/diag/DIAG.json
