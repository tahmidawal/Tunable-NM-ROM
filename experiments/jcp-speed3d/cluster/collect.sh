#!/bin/bash
# jcp-speed3d: pull a finished attempt into runs/<attempt>/archive (logs, output, manifests, provenance; not refs, cache,
# tmp), verify OUTPUTS.sha256, record files > 50 MB in LARGE_FILES.sha256, then delete the cluster job directory.
set -euo pipefail
CM="${JCP_SSH_CM:-/tmp/jcpspeed-cm-%r@%h}"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=900 -o ConnectTimeout=30"
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-speed3d
L="$ROOT/experiments/jcp-speed3d/runs/$A/archive"
mkdir -p "$L"
$SSH tufts-login "grep -q ALL-DONE $NS/$A/logs/*.out" || { echo "job not finished with ALL-DONE"; exit 2; }
rsync -e "$SSH" -a --exclude refs --exclude cache --exclude tmp --exclude experiments "tufts-login:$NS/$A/" "$L/"
(cd "$L" && sha256sum -c OUTPUTS.sha256 --quiet && echo "outputs verified")
(cd "$ROOT/experiments/jcp-speed3d" && find "runs/$A" -type f -size +50M -print0 | xargs -0 -r sha256sum >> LARGE_FILES.sha256 && sort -u -o LARGE_FILES.sha256 LARGE_FILES.sha256)
$SSH tufts-login "rm -rf $NS/$A"
echo "pulled to $L; cluster directory removed"
