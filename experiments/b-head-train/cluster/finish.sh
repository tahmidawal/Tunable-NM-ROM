#!/bin/bash
# Collect, audit, preserve and clean up one completed attempt, in one place so the
# sequence cannot drift between the two jobs. Run from the worktree root.
#
#   cluster/finish.sh train <attempt>
#   cluster/finish.sh eval  <attempt> <train-attempt>
set -euo pipefail
KIND=$1
ATTEMPT=$2
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CELL="$ROOT/experiments/b-head-train"
PY=/home/tahmid/Dev/.venv/bin/python
NS=/cluster/tufts/paralab/tawal01/b_head_train_20260916
cd "$ROOT"

"$PY" "$CELL/cluster/collect.py" "$ATTEMPT"
OUT="$CELL/runs/$ATTEMPT/archive/output"

if [ "$KIND" = train ]; then
  "$PY" "$CELL/cluster/capture_draws.py" "$OUT/result.json" --out "$CELL/artifacts/$ATTEMPT/draws.npz"
  "$PY" "$CELL/audit_train.py" "$OUT/result.json" --draws "$CELL/artifacts/$ATTEMPT/draws.npz" \
    --checkpoints "$OUT" --out "$CELL/checks/$ATTEMPT-audit.json"
else
  TRAIN=$3
  "$PY" "$CELL/audit_eval.py" "$OUT/result.json" \
    --fields "$OUT" \
    --abl01 "$ROOT/experiments/head-ablation/artifacts/abl01/result.json" \
    --train "$CELL/runs/$TRAIN/archive/output/result.json" \
    --out "$CELL/checks/$ATTEMPT-audit.json"
fi

"$PY" "$CELL/cluster/preserve_archive.py" "$ATTEMPT"
cp "$CELL/checks/$ATTEMPT-audit.json" "$CELL/artifacts/$ATTEMPT/audit.json"
ssh tufts-login "rm -rf $NS/$ATTEMPT; ls $NS"
echo "FINISHED $ATTEMPT"
