#!/bin/bash
# imp-fronts Phase B training jobs: stage one job directory locally and (with --push) rsync it to the cluster.
# Never submits.
#
#   bash experiments/imp-fronts/cluster/train/stage_job.sh <smA|smB|smC|trfid|trcov|trbnd> [--push]
#
# Local:  experiments/imp-fronts/runs/train/<job>/ (refuses if it exists) =
#           stage_arm.sh <arm> tree (experiments/imp-fronts/train2d/, PROVENANCE rewritten for the arm) + ARM-STAGED.json
#           + run.sbatch (= cluster/train/<job>.sbatch) + train_body.sh + phase_times.py + smoke_datagen.py + incumbent/burgers2d_film.py + COMMIT.txt + MANIFEST.sha256
#         Every staged source must equal its blob at HEAD (commit first), so the job runs a committed tree.
# --push: refuses if /cluster/tufts/paralab/tawal01/imp-fronts/<job> exists on tufts-login; rsyncs; verifies
#         MANIFEST.sha256 remotely. Then submit by hand (duplicate-submit rule: squeue before and after):
#           ssh tufts-login 'squeue -u $USER'; ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/imp-fronts/<job> && sbatch run.sbatch'
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <smA|smB|smC|trfid|trcov|trbnd> [--push]" >&2; exit 2; }
JOB=$1; PUSH=${2:-}
case "$JOB" in smA|smB|smC|trfid|trcov|trbnd) ;; *) echo "unknown job $JOB" >&2; exit 2;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"           # experiments/imp-fronts/cluster/train
LANE="$(cd "$HERE/../.." && pwd)"                               # experiments/imp-fronts
REPO="$(git -C "$LANE" rev-parse --show-toplevel)"
NS=/cluster/tufts/paralab/tawal01/imp-fronts
ARM=$(grep -oP '(?<=export JOB='"$JOB"' ARM=)\w+' "$HERE/$JOB.sbatch")
[ -n "$ARM" ] || { echo "no ARM in $JOB.sbatch" >&2; exit 2; }
LOCAL="$LANE/runs/train/$JOB"
[ ! -e "$LOCAL" ] || { echo "$LOCAL exists: refusing" >&2; exit 3; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain -- "$LANE/train2d" "$HERE" "$LANE/tests/test_arm_c.py")
[ -z "$DIRTY" ] || { echo "uncommitted changes under train2d/ or cluster/train/: commit first" >&2; echo "$DIRTY" >&2; exit 4; }
bash "$LANE/train2d/stage_arm.sh" "$ARM" "$LOCAL"
cp "$HERE/$JOB.sbatch" "$LOCAL/run.sbatch"
cp "$HERE/train_body.sh" "$HERE/phase_times.py" "$HERE/smoke_datagen.py" "$LOCAL/"
mkdir -p "$LOCAL/incumbent"; cp "$LANE/train2d/stageA/deps/burgers2d_film.py" "$LOCAL/incumbent/"   # K1 generator for the paired timing
echo "c521b36a9bae5471f0f3c1644d03451109c84688cf3808f9c5ea6ffea18493e4  $LOCAL/incumbent/burgers2d_film.py" | sha256sum -c --quiet
if [ "$JOB" = smC ]; then
  # SM-c extras (arm (c) identity tests + timing): the lane's train2d/ as committed (arm sources + original sep_common),
  # tests/test_arm_c.py, the K1 bank (jcp-robust s1, sha-checked), POL-K1's FI|4 scored run and the FI|4 reference
  cp "$HERE/smoke_bnd_timing.py" "$LOCAL/"
  mkdir -p "$LOCAL/lane/tests" "$LOCAL/lane/k1"
  git -C "$REPO" archive "$COMMIT" "experiments/imp-fronts/train2d" | tar -x -C "$LOCAL/lane" --strip-components=2
  cp "$LANE/tests/test_arm_c.py" "$LOCAL/lane/tests/"
  K1D=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/inputs/banks/s1
  cp "$K1D/ckpt.pkl" "$K1D/rotation_R512.npz" "$LOCAL/lane/k1/"
  ( cd "$LOCAL/lane/k1" && printf '%s  ckpt.pkl\n%s  rotation_R512.npz\n' \
      244e1107c33ae38a23d17517fc198d7f2f307f12d2bb20006d3ed7aada188954 73bb603b48f2883dfb7b40809cc67ceab86486d38dc23f2deea282356294af4d \
      | sha256sum -c --quiet )
  "${PY:-/home/tahmid/Dev/.venv/bin/python}" -I - "$LANE" "$LOCAL/lane" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
lane, dst = Path(sys.argv[1]), Path(sys.argv[2])
pol = Path('runs/rom/POL-K1-out/output/fpolicy_pol1')
case = next(c for c in json.loads((lane / pol / 'progress.json').read_text())['cases'] if c['case'] == 'FI|4')
coh, j, lev, rule, f = case['scored'].split('|')
fn = f"{coh}__{j}__{lev}__{rule}__{f.replace('.', 'p').rstrip('0').rstrip('p') if '.' in f else f}.npz"
for rel in (pol / 'result.json', pol / 'progress.json', pol / 'runs' / fn, Path('runs/refT/pulled/refslim/2d/FI/ST/c004.npz')):
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lane / rel, dst / rel)
print('SM-c inputs staged:', fn)
PYEOF
fi
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
