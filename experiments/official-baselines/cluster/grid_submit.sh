#!/usr/bin/env bash
# One CROM validation-grid chunk per job: grid_submit.sh <decoder> <arm> <setting> <a:b> [dtype] [time]
set -euo pipefail
DEC=$1; ARM=$2; SET=$3; CASES=$4; DT=${5:-float64}; TIME=${6:-12:00:00}
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/submit.sh" "g-$DEC-$ARM-$SET-$DT-${CASES/:/_}" "$TIME" "a100-80G" 48G \
  "set +e; export EXT; PYTHONPATH=\$JOB/code/crom \$CROM_PY code/crom/online_ours.py grid --weights \$NS/weights/$DEC --name $DEC --arm $ARM --setting $SET --dtype $DT --data \$DATA --out grid --cases $CASES 2>&1 | grep -E 'GRID|Error|error|Traceback' | cut -c1-600; mkdir -p \$NS/grid && cp grid/*.json \$NS/grid/"
