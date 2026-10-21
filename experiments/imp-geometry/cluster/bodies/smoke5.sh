OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 1200 "$@" > $OUT/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; grep -E '^[a-z_0-9+]+: [0-9.]+s' $OUT/$name.log | tail -30; tail -3 $OUT/$name.log; }
for s in pod fd sem bank; do run probe_L_S2_$s $PY probe.py --out $OUT/probe --dom L --lev S2 --sections $s; done
run probe_H_S2_sem $PY probe.py --out $OUT/probe --dom H --lev S2 --sections sem
run probe_L_S1_ref $PY probe.py --out $OUT/probe --dom L --lev S1 --sections ref
