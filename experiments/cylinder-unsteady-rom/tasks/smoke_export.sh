set -euo pipefail
"$PY" rom_export.py --manifest tasks/manifest_smoke.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export
cat results/export/EXPORT.json | head -8
"$PY" rom_r2.py --export results/export --case dev_smoke --out results/R2-FLOORS-smoke.json
"$PY" rom_r4.py --export results/export --manifest tasks/manifest_smoke.json --case dev_smoke --dt 0.00125 \
  --arms '[{"name":"pod_r8","kind":"pod","r":8},{"name":"frozen_bank_r4","kind":"bank","r":4,"key":"frozen"}]' --out results/r4
