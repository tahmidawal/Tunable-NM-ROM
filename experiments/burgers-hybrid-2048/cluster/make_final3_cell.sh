#!/usr/bin/env bash
# Stage the prospectively repaired, untouched-seed N=2048 primary panel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="$(cat <<'COMMAND_EOF'
$PY bh_2048_final.py ../out/final.json
SOURCE_SHA=$(sha256sum bh_2048_final.py | cut -d' ' -f1)
$PY bh_2048_audit.py ../out/final.json selection/dynamic_selection_choice.json selection/dynamic_pair_audit.json "$BH_COMMIT" "$SOURCE_SHA" ../out/final-audit.json
test -s ../out/final-audit.json
echo FINAL-AUDIT-DONE
COMMAND_EOF
)"
"$HERE/make_cell.sh" \
  ctol_hybb2048_final3 \
  256G \
  06:00:00 \
  "NS=2048 FOM_TAUS=1e-6,1e-8,1e-10 LINEAR_TOLS=1e-2,1e-4,1e-5 TEST_SEED=20260830 DRAW_COUNT=16 TEST_START=0 N_CASES=4 PAIR_BLOCKS=6 BURN_S=3 REFERENCE_RESIDUAL_GATE=1e-11 MAX_NEWTON=25 SELECTION_JSON=selection/dynamic_selection_choice.json PAIR_AUDIT_JSON=selection/dynamic_pair_audit.json BH_2048_SMOKE=0" \
  "$COMMAND" \
  h200
