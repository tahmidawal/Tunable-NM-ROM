OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 1500 "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -8 $OUT/$name.log; }
run inter_v4a $PY interaction.py --out $OUT/interaction --n 256 --ncases 16 --nsteps 2000 --variant v4a --T 0.25
for d in L H B; do run controls_geom_$d $PY controls.py --out $OUT/geom --dom $d --geometry-only; done
run probe_L_S2 $PY probe.py --out $OUT/probe --dom L --lev S2
