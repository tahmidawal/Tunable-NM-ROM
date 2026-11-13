# G5 R2 (DESIGN-G5 5.4): export on Q (order 12, P4/P3) + projection floors on development Re 95.
set -euo pipefail
"$PY" rom_export.py --manifest tasks/manifest_production.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export
head -12 results/export/EXPORT.json
"$PY" rom_r2.py --export results/export --case re95 --out results/R2-FLOORS.json
