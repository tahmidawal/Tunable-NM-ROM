#!/bin/bash
# jcp-mechanism: pull a finished job (logs, OUTPUTS.sha256, code/output) into runs/<job>/, verify every output's sha256
# locally, then delete the cluster job directory (CLAUDE.md: do not park outputs on the group share).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcpmech
LOCAL="$ROOT/experiments/jcp-mechanism/runs/$A"
ssh -n tufts-login "squeue -u \$USER -h -o '%j' | grep -qx jm_$A" && { echo "jm_$A still queued/running"; exit 3; }
ssh -n tufts-login "grep -q ALL-DONE $NS/$A/logs/*.out" || { echo "no ALL-DONE in the log: not collecting"; exit 4; }
rsync -a "tufts-login:$NS/$A/logs/" "$LOCAL/logs/"
rsync -a "tufts-login:$NS/$A/OUTPUTS.sha256" "$LOCAL/"
rsync -a "tufts-login:$NS/$A/code/output/" "$LOCAL/code/output/"
(cd "$LOCAL" && sha256sum -c OUTPUTS.sha256 --quiet) && echo "outputs verified"
ssh -n tufts-login "rm -rf $NS/$A" && echo "removed $NS/$A"
