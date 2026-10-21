OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 600 "$@" > $OUT/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; grep -E '^[a-z_0-9+ ]+: [0-9.]+s|bank sub' $OUT/$name.log | tail -12; tail -2 $OUT/$name.log; }
run probe_bankstep_S2 $PY probe.py --out $OUT/probe --dom H --lev S2 --sections bankstep
run probe_ecsw_S2 $PY probe.py --out $OUT/probe --dom L --lev S2 --sections ecsw
