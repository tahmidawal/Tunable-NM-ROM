#!/usr/bin/env bash
# Stage the bounded batch-topology repair of the diagnostic-only N=2048 job.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="$(cat <<'COMMAND_EOF'
$PY bh_2048_reference_diag.py ../out/diagnostic.json
$PY bh_2048_reference_diag_check.py ../out/diagnostic.json ../out/producer-check.json
test -s ../out/producer-check.json
echo PRODUCER-CHECK-DONE
COMMAND_EOF
)"
"$HERE/make_cell.sh" \
  ctol_hybb2048_diagnostic2 \
  96G \
  02:00:00 \
  "REFERENCE_DIAGNOSTIC=1" \
  "$COMMAND" \
  h200
