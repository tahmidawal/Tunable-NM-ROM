#!/usr/bin/env bash
# Create the exact frozen H200 stage for the diagnostic-only N=2048 job.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/make_cell.sh" \
  ctol_hybb2048_diagnostic1 \
  96G \
  02:00:00 \
  "REFERENCE_DIAGNOSTIC=1" \
  '$PY bh_2048_reference_diag.py ../out/diagnostic.json && $PY bh_2048_reference_diag_check.py ../out/diagnostic.json ../out/producer-check.json' \
  h200
