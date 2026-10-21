#!/bin/bash
# Submit a staged imp-varcoef job. Under the campaign lock (fail-closed squeue): account < 8, lane < 2, the job directory
# must not exist yet; then rsync into a fresh directory, manifest check and sbatch -- all inside the same lock.
# squeue before/after; exactly one job per directory; ledger record (atomic).   usage: bash cluster/submit.sh <job>
set -euo pipefail
JOB=$1
LANE=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef
NS=/cluster/tufts/paralab/tawal01/imp-varcoef
SRC=$LANE/runs/$JOB
[ -f "$SRC/run.sbatch" ] || { echo "not staged: $JOB"; exit 2; }
[ ! -f "$SRC/JOBID" ] || { echo "already submitted: $(cat $SRC/JOBID)"; exit 3; }
/home/tahmid/Dev/.venv/bin/python $LANE/ledger.py update >/dev/null 2>&1 || true
/home/tahmid/Dev/.venv/bin/python $LANE/ledger.py check "$JOB" ${CAP:-8} || { echo "GPU-hour cap: refused"; exit 75; }
H="ssh -o ConnectTimeout=20 tufts-login"
echo "squeue before:"; timeout 60 $H 'squeue -u tawal01 -h -o "%i %j %T %M"'
timeout 60 $H "mkdir -p $NS/.incoming"
STAGE=$NS/.incoming/$JOB.$$
timeout 900 rsync -a "$SRC/" "tufts-login:$STAGE/"
timeout 300 $H "cd $STAGE && sha256sum -c MANIFEST.sha256 --quiet && echo manifest-ok"
out=$(timeout 700 $H "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c '
  q=\$(squeue -u tawal01 -h -o %j) || exit 76
  n=\$(printf \"%s\n\" \"\$q\" | grep -c . || true); [ \$n -lt 8 ] || exit 75
  nl=\$(printf \"%s\n\" \"\$q\" | grep -c ^ivc_ || true); [ \$nl -lt 2 ] || exit 74
  [ ! -e $NS/$JOB ] || exit 73
  mv $STAGE $NS/$JOB && cd $NS/$JOB && sbatch run.sbatch'") || { rc=$?; echo "submission refused rc=$rc (75 account cap, 74 lane cap, 73 dir exists, 76 squeue failed)"; exit $rc; }
echo "$out"
jid=$(echo "$out" | awk '/Submitted batch job/{print $4}')
[ -n "$jid" ] || { echo "no job id"; exit 4; }
echo "$jid" > "$SRC/JOBID"
/home/tahmid/Dev/.venv/bin/python $LANE/ledger.py submit "$JOB" "$jid"
echo "squeue after:"; timeout 60 $H 'squeue -u tawal01 -h -o "%i %j %T %M"'
cnt=$(timeout 60 $H "squeue -u tawal01 -h -o '%j'" | grep -c "^ivc_${JOB}\$" || true)
[ "$cnt" -eq 1 ] || echo "WARNING: $cnt jobs named ivc_$JOB in the queue"
echo "submitted $JOB as $jid"
