# Rebuttal push, one training arm (PROTOCOL.md step 2): train -> development projections -> development
# rollouts (bank at full rank and PCA prefixes, POD at the same ranks) -> diagnosis.  Development Re 95 only.
# Expects RANK, TRAIN_ARGS, ARMS (and optionally EX) from the arm script.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=${EX:-$NS/r2-export01/results/export}
M=tasks/manifest_production.json
"$PY" bank_train2.py --export "$EX" --rank "$RANK" --out results/train $TRAIN_ARGS
"$PY" rom_project.py --manifest $M --out-dir results/proj --export "$EX" \
  --bank-eval results/train/trained_bank_eval.npz --tag trained --case re95
"$PY" rom_r4.py --export "$EX" --manifest $M --case re95 --dt 0.0003125 --uncertainty inputs/R1DE.json --out results/r4 \
  --trained-bank results/train/trained_bank_eval.npz --trained-proj results/proj/proj_re95_trained.npz --arms "$ARMS"
"$PY" bank_diag.py --export "$EX" --manifest $M --case re95 --bank-eval results/train/trained_bank_eval.npz \
  --bank-proj results/proj/proj_re95_trained.npz --rank "$RANK" --out results/diag/DIAG.json
