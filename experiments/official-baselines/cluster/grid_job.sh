#!/usr/bin/env bash
# One job running a list of CROM validation settings in order: grid_job.sh <jobname> <time> <dtype> <dec:arm:setting>...
set -euo pipefail
NAME=$1; TIME=$2; DT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/submit.sh" "$NAME" "$TIME" "a100-80G" 48G "export EXT NS DATA JOB CROM_PY; bash code/crom/grid_driver.sh $DT $*"
