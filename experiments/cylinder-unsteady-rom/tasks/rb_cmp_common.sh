# Rebuttal push, PROTOCOL.md amendment A6: paired cost comparison in ONE allocation (CPU, 16 threads):
# ROM timings -> FE ladder (reps 1) -> accuracy-matched FE chosen by the comparator rule -> that entry timed
# (reps 5) -> ROM timings again.  Expects CASE, EXPORT, BANK, PROJ, TARMS, DTS, FE_LADDER, FE_FLAGS.
"$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
  --out results/timing/ROM-before.json --trained-bank $BANK --trained-proj $PROJ
"$PY" fe_ladder.py --manifest $M --case $CASE $FE_FLAGS --out results/ladder --entries "$FE_LADDER" --reps 1
CHOSEN=$("$PY" select_fe.py --ladder results/ladder/LADDER-$CASE.json --out results/ladder/SELECT-$CASE.json)
echo "accuracy_matched_fe=$CHOSEN"
if [ "$CHOSEN" != "[]" ]; then
  "$PY" fe_ladder.py --manifest $M --case $CASE $FE_FLAGS --out results/ladder_timed --entries "$CHOSEN" --reps 5
fi
"$PY" rom_time2.py --export $EXPORT --manifest $M --case $CASE --dts "$DTS" --arms "$TARMS" --reps 5 \
  --out results/timing/ROM-after.json --trained-bank $BANK --trained-proj $PROJ
