#!/bin/bash
# panel_submit.sh <problem> <n> <train job> <gres> <constraint|-> <mem> <jax fraction>
# Pull the cell's training records (no checkpoints), write configs/panel_<cell>.json, stage pn_<cell>,
# copy the checkpoints cluster-side from the training job dir, submit (squeue before/after in submit.sh).
set -euo pipefail
P="$1"; N="$2"; TJ="$3"; GRES="$4"; CONS="$5"; MEM="$6"; JF="$7"
NS=/cluster/tufts/paralab/tawal01/opsall_20260924/l3d
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELL=$P$N; JOB=pn_$CELL
mkdir -p "$HERE/runs/$TJ/output"
rsync -a --exclude='*.pt' --exclude='*.tmp' "tufts-login:$NS/$TJ/output/$CELL" "$HERE/runs/$TJ/output/"
/home/tahmid/Dev/.venv/bin/python "$HERE/make_panel_config.py" "$P" "$N" "runs/$TJ/output/$CELL"
"$HERE/cluster/stage.sh" "$JOB"
for a in $(ls "$HERE/runs/$TJ/output/$CELL"); do
  ssh tufts-login "test -f $NS/$TJ/output/$CELL/$a/best.pt && mkdir -p $NS/$JOB/code/checkpoints/$a && cp $NS/$TJ/output/$CELL/$a/best.pt $NS/$JOB/code/checkpoints/$a/best.pt || true"
done
"$HERE/cluster/submit.sh" "$JOB" "$GRES" "$CONS" "$MEM" panel.sbatch "configs/panel_$CELL.json" "$JF"
