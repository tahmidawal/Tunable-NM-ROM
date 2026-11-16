#!/usr/bin/env bash
# CROM validation grid driver (PROTOCOL.md 5.2), run inside a job from submit.sh (needs CROM_PY, NS, DATA, EXT, JOB).
#   grid_driver.sh <dtype> <decoder:arm:setting> [...]
# Per setting: cases 0:8 first; if 4 x (their wall time) > 6 GPU-hours the setting stays at 8 cases (labelled
# partial by select_frozen), otherwise cases 8:32 follow. Chunk JSONs go straight to $NS/grid (skip if present).
set -uo pipefail
DT=$1; shift
mkdir -p "$NS/grid"
for spec in "$@"; do
  IFS=: read -r DEC ARM SET <<< "$spec"
  run() { PYTHONPATH=$JOB/code/crom "$CROM_PY" "$JOB/code/crom/online_ours.py" grid --weights "$NS/weights/$DEC" --name "$DEC" --arm "$ARM" \
            --setting "$SET" --dtype "$DT" --data "$DATA" --out "$NS/grid" --cases "$1" 2>&1 | grep -E 'GRID|SKIP|Error|error|Traceback' | cut -c1-700; }
  run 0:8
  F="$NS/grid/$DEC-$ARM-$SET-$DT-0_8.json"
  [ -f "$F" ] || { echo "MISSING $F"; continue; }
  PROJ=$("$CROM_PY" -c "import json,sys; d=json.load(open('$F')); print(4*sum(d['ms_per_case'])/1e3)")
  echo "PROJECT $spec 32-case seconds=$PROJ"
  if "$CROM_PY" -c "import sys; sys.exit(0 if $PROJ <= 6*3600 else 1)"; then run 8:32; else echo "PARTIAL $spec (projected > 6 GPU-h)"; fi
done
echo DRIVER-DONE
