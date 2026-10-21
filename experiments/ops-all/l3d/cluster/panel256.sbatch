#!/bin/bash
# 256^3 panel, chained after tr_256c (sbatch --dependency=afterany:<tr_256c>). Builds the cell's operator list on the
# cluster from tr_256c's result.json files (U-Net-s, Transolver-s) and the first attempt's failure records
# (first_attempt_256/: FNO-s, DeepONet-s), copies the checkpoints, writes the config, then runs the panel.
# sbatch ... panel256.sbatch <problem> <jax fraction>
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
set -uo pipefail
P="$1"; JFRAC="${2:-0.6}"; CELL=${P}256
TASK_ROOT="$PWD"; NS=/cluster/tufts/paralab/tawal01/opsall_20260924/l3d
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
cd code
mkdir -p merged/$CELL
cp -r first_attempt_256/$CELL/* merged/$CELL/
X=${P:0:1}
for aj in unet-s:tr_256c tsol-s:tr_256c fno-s:tr_256f$X don-s:tr_256d$X; do
  a=${aj%%:*}; j=${aj#*:}
  mkdir -p merged/$CELL/$a
  # the whole-GPU attempt's record replaces the first attempt's failure record when it exists
  cp $NS/$j/output/$CELL/$a/result.json merged/$CELL/$a/ 2>/dev/null && echo "record $a from $j" || echo "no result.json for $a in $j (kept: $(ls merged/$CELL/$a))"
  if [ -f $NS/$j/output/$CELL/$a/best.pt ]; then mkdir -p checkpoints/$a && cp $NS/$j/output/$CELL/$a/best.pt checkpoints/$a/; fi
done
"$PY" make_panel_config.py "$P" 256 "merged/$CELL"
cd "$TASK_ROOT"
exec bash code/cluster/panel.sbatch "configs/panel_$CELL.json" "$JFRAC"
