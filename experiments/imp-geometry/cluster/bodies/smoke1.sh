OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 1200 "$@" > $OUT/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -25 $OUT/$name.log; }
run interaction $PY interaction.py --out $OUT/interaction --n 256 --ncases 8 --nsteps 2000
for d in L H B; do run controls_geom_$d $PY controls.py --out $OUT/geom --dom $d --geometry-only; done
run cell_L $PY run_cell.py --dom L --fam wide --lev S1 --out $OUT/cellL --smoke --stages ref,data,bank,ladder,time
run controls_cell_L $PY controls.py --out $OUT/cellL --dom L --fam wide --lev S1 --smoke
run cell_B $PY run_cell.py --dom B --fam wide --lev S1 --out $OUT/cellB --smoke --stages ref,data,ladder
