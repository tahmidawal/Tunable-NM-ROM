OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 900 "$@" > $OUT/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -12 $OUT/$name.log; }
for v in v0 v2; do for T in 0.25 0.5; do run inter_${v}_$T $PY interaction.py --out $OUT/interaction --n 256 --ncases 16 --nsteps $(python3 -c "print(int(8000*$T))") --variant $v --T $T; done; done
for d in L H B; do run controls_geom_$d $PY controls.py --out $OUT/geom --dom $d --geometry-only; done
run cell_L $PY run_cell.py --dom L --fam wide --lev S1 --out $OUT/cellL --smoke --stages ref,data,bank,ladder,time
run controls_cell_L $PY controls.py --out $OUT/cellL --dom L --fam wide --lev S1 --smoke
