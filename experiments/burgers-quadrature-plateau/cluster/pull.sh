#!/bin/bash
# Pull one finished job: rsync to runs/<job>/ (untracked), verify OUTPUTS.sha256 locally, copy result.json + logs into
# results/<job>/ (committed), record sha256 of every field file in results/<job>/FIELDS.sha256, then delete the remote dir
# only if verification passed.   bash cluster/pull.sh <job>
set -euo pipefail
JOB="$1"; [[ "$JOB" =~ ^[a-zA-Z0-9]+$ ]]
cd "$(dirname "$0")/.."
NS=/cluster/tufts/paralab/tawal01/bqplateau-20261105
mkdir -p runs/$JOB results/$JOB
rsync -a --exclude experiments --exclude cache --exclude tmp "tufts-login:$NS/$JOB/" runs/$JOB/
(cd runs/$JOB && sha256sum -c OUTPUTS.sha256 --quiet) && echo "OUTPUTS.sha256 verified"
grep -q ALL-DONE runs/$JOB/logs/*.out
cp runs/$JOB/output/result.json results/$JOB/result.json
cp runs/$JOB/logs/*.out runs/$JOB/logs/*.err runs/$JOB/run.sbatch runs/$JOB/COMMIT.txt results/$JOB/
cp runs/$JOB/OUTPUTS.sha256 results/$JOB/OUTPUTS.sha256
ssh tufts-login "rm -rf $NS/$JOB" && echo "remote $NS/$JOB deleted"
