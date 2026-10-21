#!/bin/bash
# imp-fronts Phase B training jobs: stage one job directory locally and (with --push) rsync it to the cluster.
# Never submits.
#
#   bash experiments/imp-fronts/cluster/train/stage_job.sh <smA|smB|trfid|trcov> [--push]
#
# Local:  experiments/imp-fronts/runs/train/<job>/ (refuses if it exists) =
#           stage_arm.sh <arm> tree (experiments/imp-fronts/train2d/, PROVENANCE rewritten for the arm) + ARM-STAGED.json
#           + run.sbatch (= cluster/train/<job>.sbatch) + train_body.sh + phase_times.py + COMMIT.txt + MANIFEST.sha256
#         Every staged source must equal its blob at HEAD (commit first), so the job runs a committed tree.
# --push: refuses if /cluster/tufts/paralab/tawal01/imp-fronts/<job> exists on tufts-login; rsyncs; verifies
#         MANIFEST.sha256 remotely. Then submit by hand (duplicate-submit rule: squeue before and after):
#           ssh tufts-login 'squeue -u $USER'; ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/imp-fronts/<job> && sbatch run.sbatch'
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <smA|smB|trfid|trcov> [--push]" >&2; exit 2; }
JOB=$1; PUSH=${2:-}
case "$JOB" in smA|smB|trfid|trcov) ;; *) echo "unknown job $JOB" >&2; exit 2;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # experiments/imp-fronts/cluster/train
LANE="$(cd "$HERE/../.." && pwd)"                               # experiments/imp-fronts
REPO="$(git -C "$LANE" rev-parse --show-toplevel)"
NS=/cluster/tufts/paralab/tawal01/imp-fronts
ARM=$(grep -oP '(?<=export JOB='"$JOB"' ARM=)\w+' "$HERE/$JOB.sbatch")
[ -n "$ARM" ] || { echo "no ARM in $JOB.sbatch" >&2; exit 2; }
LOCAL="$LANE/runs/train/$JOB"
[ ! -e "$LOCAL" ] || { echo "$LOCAL exists: refusing" >&2; exit 3; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain -- "$LANE/train2d" "$HERE")
[ -z "$DIRTY" ] || { echo "uncommitted changes under train2d/ or cluster/train/: commit first" >&2; echo "$DIRTY" >&2; exit 4; }
bash "$LANE/train2d/stage_arm.sh" "$ARM" "$LOCAL"
cp "$HERE/$JOB.sbatch" "$LOCAL/run.sbatch"
cp "$HERE/train_body.sh" "$HERE/phase_times.py" "$LOCAL/"
echo "$COMMIT" > "$LOCAL/COMMIT.txt"
mkdir -p "$LOCAL/logs"
( cd "$LOCAL" && find . -type f ! -name MANIFEST.sha256 ! -path './logs/*' -printf '%P\n' | sort | xargs sha256sum > MANIFEST.sha256 )
echo "staged $JOB (arm $ARM, commit $COMMIT) -> $LOCAL ($(wc -l < "$LOCAL/MANIFEST.sha256") files)"
if [ "$PUSH" = "--push" ]; then
  timeout 120 ssh -o ConnectTimeout=20 tufts-login "test ! -e $NS/$JOB && mkdir -p $NS/$JOB/logs"
  rsync -a "$LOCAL/" "tufts-login:$NS/$JOB/"
  timeout 300 ssh -o ConnectTimeout=20 tufts-login "cd $NS/$JOB && sha256sum -c MANIFEST.sha256 --quiet && echo REMOTE-MANIFEST-OK"
  echo "pushed to tufts-login:$NS/$JOB (not submitted)"
fi
