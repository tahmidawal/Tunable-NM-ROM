#!/bin/bash
# Pull one finished job's small outputs into results/<job>/ (committed), verify OUTPUTS.sha256, then delete the remote dir.
#   bash cluster/pull.sh <job>
set -euo pipefail
JOB="$1"; [[ "$JOB" =~ ^[a-zA-Z0-9]+$ ]]
cd "$(dirname "$0")/.."
NS=/cluster/tufts/paralab/tawal01/tcross-20261115
D=/tmp/claude-1002/-home-tahmid-Dev-Tunable-NM-ROM/8a27696f-3021-599f-9eec-6de38966c9fa/scratchpad/tcx_pull_$JOB
rm -rf "$D"; mkdir -p "$D" results/$JOB
rsync -a --exclude experiments --exclude cache --exclude tmp "tufts-login:$NS/$JOB/" "$D/"
(cd "$D" && sha256sum -c OUTPUTS.sha256 --quiet) && echo "OUTPUTS.sha256 verified"
grep -q ALL-DONE "$D"/logs/*.out
cp "$D"/output/result.json "$D"/logs/*.out "$D"/logs/*.err "$D"/run.sbatch "$D"/COMMIT.txt "$D"/OUTPUTS.sha256 results/$JOB/
rm -rf "$D"
ssh -o BatchMode=yes tufts-login "rm -rf $NS/$JOB" && echo "remote $NS/$JOB deleted"
