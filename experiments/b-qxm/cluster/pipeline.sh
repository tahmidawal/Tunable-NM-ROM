#!/bin/bash
# Post-job pipeline for one attempt: collect (checksum-verified) -> independent NumPy audit ->
# Git chunks -> explicit remote delete of the exact attempt directory. Stops at the first failure.
#   bash cluster/pipeline.sh <attempt>
set -euo pipefail
A="$1"; CELL=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-qxm/experiments/b-qxm
PY=/home/tahmid/Dev/.venv/bin/python
NS=/cluster/tufts/paralab/tawal01/b_qxm_20260917
cd "$CELL"
$PY cluster/collect.py "$A"
grep -q "ALL-DONE" runs/$A/archive/logs/*.out
$PY audit_xm.py runs/$A/archive/output/result.json --fields runs/$A/archive/output --out checks/$A-audit.json
cp checks/$A-audit.json runs/$A/audit.json
$PY cluster/preserve_archive.py "$A"
ssh tufts-login "rm -rf $NS/$A && ls $NS"
echo "PIPELINE DONE $A"
