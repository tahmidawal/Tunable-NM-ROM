#!/bin/bash
# submit.sh <job> <gres> <constraint|-> <mem> <sbatch file> <args...>   (after stage.sh <job>)
# squeue before and after; refuses if a job with this name is already queued.
set -euo pipefail
JOB="$1"; GRES="$2"; CONS="$3"; MEM="$4"; SB="$5"; shift 5
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/l3d
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HERE/runs/$JOB"
echo "--- squeue BEFORE"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"' | tee "$HERE/runs/$JOB/QUEUE-BEFORE.log"
if grep -q " opsl3d_$JOB " "$HERE/runs/$JOB/QUEUE-BEFORE.log"; then echo "duplicate job name queued"; exit 1; fi
CF="${DEP:+--dependency=$DEP}"; [[ "$CONS" != "-" ]] && CF="$CF --constraint=$CONS"
ID=$(ssh tufts-login "cd $NS/$JOB && test ! -e JOBID && sbatch --parsable --job-name=opsl3d_$JOB --gres=$GRES $CF --mem=$MEM \
  --output=$NS/$JOB/logs/%j.out --error=$NS/$JOB/logs/%j.err code/cluster/$SB $* && true")
ssh tufts-login "echo $ID > $NS/$JOB/JOBID"
echo "$ID" > "$HERE/runs/$JOB/JOBID"
echo "submitted $JOB -> $ID"
sleep 3
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.28j %.8T %.10M %R"' | tee "$HERE/runs/$JOB/QUEUE-AFTER.log"
[ "$(grep -c " opsl3d_$JOB " "$HERE/runs/$JOB/QUEUE-AFTER.log")" -le 1 ] || { echo "DUPLICATE SUBMISSION"; exit 1; }
echo "$JOB $ID $GRES $CONS $MEM $SB $*" >> "$HERE/runs/JOBS.txt"
