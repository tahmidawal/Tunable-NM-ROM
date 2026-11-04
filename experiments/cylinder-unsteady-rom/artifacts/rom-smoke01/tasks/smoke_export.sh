set -euo pipefail
/usr/bin/time -v "$PY" rom_export.py --manifest tasks/manifest_smoke.json --bank inputs/run09_bank.npz \
  --bank-sha256 645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60 --out results/export 2> results/export_time.txt
tail -25 results/export_time.txt | grep -E 'Elapsed|Maximum resident'
"$PY" rom_r2.py --export results/export --case dev_smoke --out results/R2-FLOORS-smoke.json
