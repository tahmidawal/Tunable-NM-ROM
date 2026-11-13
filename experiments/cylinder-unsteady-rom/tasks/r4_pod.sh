# G5 R4 (DESIGN-G5 5.4): development rollouts (Re 95) for the arms not depending on R3.
set -euo pipefail
EX=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/r2-export01/results/export
sha256sum -c <(grep -E 'quadrature.npz|basis.npz|pod.npz' inputs/EXPORT-SHA256SUMS | sed "s#  ./results/export/#  $EX/#")
"$PY" rom_r4.py --export "$EX" --manifest tasks/manifest_production.json --case re95 --dt 0.0003125 \
  --uncertainty inputs/R1DE.json --out results/r4 \
  --arms '[{"name":"pod_r8","kind":"pod","r":8},{"name":"pod_r16","kind":"pod","r":16},{"name":"pod_r24","kind":"pod","r":24},{"name":"frozen_bank_r4","kind":"bank","r":4,"key":"frozen"}]'
