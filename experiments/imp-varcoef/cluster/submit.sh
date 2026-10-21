#!/bin/bash
# Submit a staged imp-varcoef job: rsync, remote manifest check, campaign lock (account < 8), lane cap (< 2), sbatch,
# squeue before/after, ledger record.   usage: bash cluster/submit.sh <job>
set -euo pipefail
JOB=$1
LANE=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef
NS=/cluster/tufts/paralab/tawal01/imp-varcoef
SRC=$LANE/runs/$JOB
[ -f "$SRC/run.sbatch" ] || { echo "not staged: $JOB"; exit 2; }
H="ssh -o ConnectTimeout=20 tufts-login"
echo "squeue before:"; timeout 60 $H 'squeue -u tawal01 -h -o "%i %j %T %M"' || true
n_lane=$(timeout 60 $H 'squeue -u tawal01 -h -o "%j"' | grep -c '^ivc_' || true)
[ "$n_lane" -lt 2 ] || { echo "lane cap reached ($n_lane)"; exit 75; }
timeout 60 $H "test ! -e $NS/$JOB" || { echo "remote dir exists: $NS/$JOB"; exit 3; }
timeout 60 $H "mkdir -p $NS"
timeout 900 rsync -a "$SRC/" "tufts-login:$NS/$JOB/"
timeout 300 $H "cd $NS/$JOB && sha256sum -c MANIFEST.sha256 --quiet && echo manifest-ok"
out=$(timeout 700 $H "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $NS/$JOB && sbatch run.sbatch'") || { rc=$?; echo "sbatch refused rc=$rc"; exit $rc; }
echo "$out"
jid=$(echo "$out" | awk '/Submitted batch job/{print $4}')
[ -n "$jid" ] || { echo "no job id"; exit 4; }
echo "$jid" > "$SRC/JOBID"
echo "squeue after:"; timeout 60 $H 'squeue -u tawal01 -h -o "%i %j %T %M"'
cnt=$(timeout 60 $H "squeue -u tawal01 -h -o '%j'" | grep -c "^ivc_${JOB}\$" || true)
[ "$cnt" -eq 1 ] || echo "WARNING: $cnt jobs named ivc_$JOB in the queue"
/home/tahmid/Dev/.venv/bin/python $LANE/ledger.py submit "$JOB" "$jid"
echo "submitted $JOB as $jid"
