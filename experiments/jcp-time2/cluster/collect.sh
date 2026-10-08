#!/bin/bash
# jcp-time2: pull a finished attempt into runs/<attempt>/archive (logs, output, manifests, provenance; not the staged
# references, cache or tmp), verify OUTPUTS.sha256 locally, record large files (> 50 MB) in LARGE_FILES.sha256, then
# delete the cluster job directory (CLAUDE.md: pull, then clean up).
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcptime2
L="$ROOT/experiments/jcp-time2/runs/$A/archive"
mkdir -p "$L"
ssh tufts-login "grep -q ALL-DONE $NS/$A/logs/*.out" || { echo "job not finished with ALL-DONE"; exit 2; }
rsync -a --exclude refs --exclude cache --exclude tmp --exclude '*.pkl' "tufts-login:$NS/$A/" "$L/"
(cd "$L" && sha256sum -c OUTPUTS.sha256 --quiet && echo "outputs verified")
(cd "$ROOT/experiments/jcp-time2" && find "runs/$A/archive" -type f -size +50M -print0 | xargs -0 -r sha256sum >> LARGE_FILES.sha256 && sort -u -o LARGE_FILES.sha256 LARGE_FILES.sha256)
ssh tufts-login "rm -rf $NS/$A"
echo "pulled to $L; cluster directory removed"
