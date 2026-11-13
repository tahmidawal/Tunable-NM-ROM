# G5 R3 (DESIGN-G5 5.4): retrain the bank at r_b = 16 (R2 rank rule) on transient POD teachers,
# then identity-bound projections of development Re 95 onto the trained bank.
set -euo pipefail
EX=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r2-export01/results/export
sha256sum -c <(grep -E 'quadrature.npz|basis.npz|pod.npz' inputs/EXPORT-SHA256SUMS | sed "s#  ./results/export/#  $EX/#")
"$PY" bank_train.py --export "$EX" --rank 16 --out results/train
"$PY" rom_project.py --manifest tasks/manifest_production.json --out-dir results/proj --export "$EX" \
  --bank-eval results/train/trained_bank_eval.npz --tag trained --case re95
