# G5 quadrature check, attempt 2 (G5-A5 item 7): order-14 export REUSING the order-12 FE modes by hash.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=$NS/r2-export01/results/export; TR=$NS/r3-train01/results; R4LOW=$NS/r4-bank-qcheck01/results/r4/R4-re95.json
sha256sum -c <(grep -E 'quadrature.npz|basis.npz|pod.npz' inputs/EXPORT-SHA256SUMS | sed "s#  ./results/export/#  $EX/#")
sha256sum -c inputs/R3-SHA256SUMS
ARMS_ALL='[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"frozen_bank_r4","kind":"bank","r":4,"key":"frozen"},{"name":"bank_r16","kind":"bank","r":16,"key":"trained"}]'
"$PY" rom_export.py --manifest tasks/manifest_production_q14.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export_q14 \
  --pod-from $EX/pod.npz --pod-sha256 56c74a805618fe5c13d6ff795aa4125d15fce791401a9e891da7abde0485bf80
"$PY" bank_train.py --export results/export_q14 --rank 16 --out results/train_q14 --eval-only $TR/train/trained_bank_params.npz
"$PY" rom_project.py --manifest tasks/manifest_production_q14.json --export results/export_q14 \
  --bank-eval results/train_q14/trained_bank_eval.npz --tag trained --case re95 --out-dir results/proj_q14
"$PY" rom_r4.py --export results/export_q14 --manifest tasks/manifest_production_q14.json --case re95 --dt 0.0003125 \
  --uncertainty inputs/R1DE.json --out results/r4_q14 --trained-bank results/train_q14/trained_bank_eval.npz \
  --trained-proj results/proj_q14/proj_re95_trained.npz --arms "$ARMS_ALL"
"$PY" rom_qcheck.py --low "$EX" --high results/export_q14 --case re95 --arms "$ARMS_ALL" \
  --trained-low $TR/train/trained_bank_eval.npz --trained-high results/train_q14/trained_bank_eval.npz \
  --trained-proj-low $TR/proj/proj_re95_trained.npz --trained-proj-high results/proj_q14/proj_re95_trained.npz \
  --r4-low $R4LOW --r4-high results/r4_q14/R4-re95.json --out results/QCHECK.json
