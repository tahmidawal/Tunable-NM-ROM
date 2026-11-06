# R4 animation frames with velocity vectors (pointwise vector error), development Re 95.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
EX=$NS/r2-export01/results/export; TR=$NS/r3-train01/results
sha256sum -c <(grep -E 'quadrature.npz|basis.npz|pod.npz' inputs/EXPORT-SHA256SUMS | sed "s#  ./results/export/#  $EX/#")
sha256sum -c inputs/R3-SHA256SUMS
"$PY" rom_frames.py --export "$EX" --manifest tasks/manifest_production.json --case re95 \
  --series-dir $NS/r4-bank-qcheck01/results/r4 --trained-bank $TR/train/trained_bank_eval.npz \
  --arms '[{"name":"pod_r16","kind":"pod","r":16},{"name":"bank_r16","kind":"bank","r":16,"key":"trained"}]' --out results/frames_re95.npz
