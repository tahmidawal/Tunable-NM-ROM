#!/bin/bash
# wait for Slurm job(s) to reach a terminal state; heartbeat between polls; overall deadline (seconds).
# usage: bash cluster/wait.sh <deadline_s> <step> <jobid> [<jobid> ...]
LANE=/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef
DL=$1; STEP=$2; shift 2; IDS="$*"
end=$(( $(date +%s) + DL ))
while [ $(date +%s) -lt $end ]; do
  all=1
  for j in $IDS; do
    s=$(timeout 60 ssh -o ConnectTimeout=20 tufts-login "sacct -j $j -X -n -o State" 2>/dev/null | head -1 | awk '{print $1}')
    case "$s" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) ;; *) all=0 ;; esac
    echo "$(date -u +%FT%TZ) $j ${s:-unknown}"
  done
  bash $LANE/hb.sh "$STEP" "$(echo $IDS | tr ' ' ',')" "" "" "$(date -u -d @$end +%FT%TZ)" "${NOTE:-waiting on slurm}" >/dev/null 2>&1
  [ $all -eq 1 ] && { echo TERMINAL; exit 0; }
  sleep 240
done
echo DEADLINE; exit 1
