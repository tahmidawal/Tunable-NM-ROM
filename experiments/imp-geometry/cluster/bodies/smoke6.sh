OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 220 "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; grep -E 'bank sub' $OUT/$name.log | tail -2; tail -2 $OUT/$name.log; }
run probe_bankstep_H_S2 $PY probe.py --out $OUT/probe --dom H --lev S2 --sections bankstep
run probe_bankstep_L_S1 $PY probe.py --out $OUT/probe --dom L --lev S1 --sections bankstep
