#!/bin/bash
# submit_cell.sh <job> <mesh> <arms> <ntrain> <nval> <jfrac> <gres> <constraint|-> <mem> <hours> [extra-ops-dirs|-] [cell|train] [dependency-jobid]
# Stages this directory's code (working tree; sha256 manifest + COPIED-FROM.json record provenance) into a
# fresh job dir of the group namespace; squeue before and after; refuses to reuse a directory.
set -euo pipefail
JOB="$1"; MESH="$2"; ARMS="$3"; NTRAIN="$4"; NVAL="$5"; JFRAC="$6"; GRES="$7"; CONS="$8"; MEM="$9"; HOURS="${10}"
EXTRA="${11:--}"; MODE="${12:-cell}"; DEP="${13:-}"
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/b3d
DEST="$NS/$JOB"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=$(ssh tufts-login 'squeue -u tawal01 -h -o "%i %j %T"')
echo "--- squeue BEFORE"; echo "$Q"
if echo "$Q" | awk '{print $2}' | grep -qx "opsb3d_$JOB"; then echo "opsb3d_$JOB already queued: refusing"; exit 3; fi
ssh tufts-login "mkdir -p $NS && mkdir $DEST && mkdir -p $DEST/logs $DEST/code" || { echo "refusing: $DEST exists"; exit 1; }
rsync -a --exclude='runs/' --exclude='__pycache__/' --exclude='*.md' --exclude='results.json' "$HERE/" "tufts-login:$DEST/code/"
( cd "$HERE" && find . -type f ! -path './runs/*' ! -path '*/__pycache__/*' ! -name '*.md' ! -name 'results.json' -print0 | sort -z | xargs -0 sha256sum ) | ssh tufts-login "cat > $DEST/MANIFEST.sha256"
git -C "$HERE" rev-parse HEAD | ssh tufts-login "cat > $DEST/COMMIT.txt"
C=""; [[ "$CONS" != "-" ]] && C="--constraint=$CONS"
[[ -n "$DEP" ]] && C="$C --dependency=afterany:$DEP"
ID=$(ssh tufts-login "cd $DEST && sbatch --parsable --job-name=opsb3d_$JOB --gres=$GRES $C --mem=$MEM --time=$HOURS:00:00 \
  --output=$DEST/logs/%j.out --error=$DEST/logs/%j.err code/cluster/cell.sbatch $MESH $ARMS $NTRAIN $NVAL $JFRAC $EXTRA $MODE")
echo "submitted job $ID into $DEST"
sleep 3
echo "--- squeue AFTER"; ssh tufts-login 'squeue -u tawal01 -o "%.10i %.24j %.8T %.10M %.12b %R"'
mkdir -p "$HERE/runs"; echo "$JOB $ID $MESH $GRES $(date -Is)" >> "$HERE/runs/JOBS.txt"
